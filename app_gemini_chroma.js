import cors from 'cors'
import express from 'express'

import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import * as dotenv from "dotenv";
dotenv.config();
console.log("Valor de GOOGLE_API_KEY:", process.env.GOOGLE_API_KEY);

import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";

import { TextLoader } from "langchain/document_loaders/fs/text";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter"

//import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { Chroma } from "@langchain/community/vectorstores/chroma";

import { loadQAStuffChain } from "langchain/chains";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";

import fs from 'fs';
import path from 'path';

import { ChromaClient } from "chromadb";


// --- Importaciones y Configuración para Google Drive API ---
import { google } from 'googleapis';
const CLIENT_ID = process.env.GOOGLE_CLIENT_ID;
const CLIENT_SECRET = process.env.GOOGLE_CLIENT_SECRET;
const REDIRECT_URI = process.env.GOOGLE_REDIRECT_URI; // O una URI de aplicación instalada para desarrollo local
const REFRESH_TOKEN = process.env.GOOGLE_REFRESH_TOKEN; // Obtenido después de la autorización inicial
const oAuth2Client = new google.auth.OAuth2(CLIENT_ID, CLIENT_SECRET, REDIRECT_URI);
oAuth2Client.setCredentials({ refresh_token: REFRESH_TOKEN });
const drive = google.drive({ version: 'v3', auth: oAuth2Client });
// --- Fin de Configuración para Google Drive ---

const app = express();
const port = 3000;
app.use(cors({ origin: ['http://localhost:4200', 'http://10.3.0.62:4200', 'http://aplicaciones.fce.unju.edu.ar', 'http://aplicaciones.unju.edu.ar'] }));

const embeddingstext = new GoogleGenerativeAIEmbeddings({
    model: "text-embedding-004", // 768 dimensiones
    taskType: TaskType.RETRIEVAL_DOCUMENT,
    title: "Document title",
});

let vectorStore; // Se declara vectorStore globalmente para que sea accesible desde cualquier endpoint






const GOOGLE_DRIVE_FOLDER_ID = process.env.GOOGLE_DRIVE_FOLDER_ID;
const LOCAL_DOCS_FOLDER = './downloaded_docs';


/**
 * @function deleteLocalFolder
 * @description Elimina los documentos locales
 * @returns {Promise<void>} Una promesa que resuelve cuando el vector store ha sido inicializado.
 */
async function deleteLocalFolder() {
    // --- Lógica para ELIMINAR EL CONTENIDO de la carpeta local antes de descargar de google drive  ---
    if (fs.existsSync(LOCAL_DOCS_FOLDER)) {
        console.log(`Borrando el contenido de la carpeta local: ${LOCAL_DOCS_FOLDER}`);
        const files = await fs.promises.readdir(LOCAL_DOCS_FOLDER); // Lee el contenido de la carpeta
        for (const file of files) {
            const filePath = path.join(LOCAL_DOCS_FOLDER, file);
            try {
                const stats = await fs.promises.lstat(filePath); // Obtiene información del elemento
                if (stats.isDirectory()) {
                    // Si es un directorio, lo borra recursivamente
                    await fs.promises.rm(filePath, { recursive: true, force: true });
                    console.log(`   - Subcarpeta eliminada: ${file}`);
                } else {
                    // Si es un archivo, lo borra
                    await fs.promises.unlink(filePath);
                    console.log(`   - Archivo eliminado: ${file}`);
                }
            } catch (err) {
                console.error(`   - Error al borrar ${filePath}:`, err);
            }
        }
        console.log(`Contenido de la carpeta '${LOCAL_DOCS_FOLDER}' borrado.`);
    } else {
        console.log(`La carpeta '${LOCAL_DOCS_FOLDER}' no existe. Creándola...`);
        fs.mkdirSync(LOCAL_DOCS_FOLDER, { recursive: true });
    }
    // --- FIN LÓGICA ELIMINAR CONTENIDO CARPETA LOCAL---
}



/**
 * @function downloadFilesFromDrive
 * @description Descarga archivos de una carpeta específica de Google Drive a una carpeta local.
 * Maneja archivos de Google Docs (exportándolos a texto plano) y otros tipos de archivos.
 * También explora subcarpetas recursivamente.
 * @param {string} folderId - El ID de la carpeta de Google Drive a descargar.
 * @param {string} destinationFolder - La ruta de la carpeta local donde se guardarán los archivos.
 * @returns {Promise<string[]>} Una promesa que resuelve con un array de las rutas de los archivos descargados.
 */
async function downloadFilesFromDrive(folderId, destinationFolder) {
    console.log("Iniciando descarga de archivos desde Google Drive...");

    // Crea la carpeta de destino si no existe
    if (!fs.existsSync(destinationFolder)) {
        fs.mkdirSync(destinationFolder, { recursive: true });
    }

    const downloadedFilePaths = [];

    // Lista los archivos y carpetas dentro de la carpeta de Google Drive
    const filesList = await drive.files.list({
        q: `'${folderId}' in parents and trashed = false`, // Busca archivos en la carpeta y que no estén en la papelera
        fields: 'files(id, name, mimeType)', // Campos a obtener de cada archivo/carpeta
    });

    const items = filesList.data.files;
    if (items.length === 0) {
        console.log(`No se encontraron archivos ni subcarpetas en la carpeta de Google Drive con ID: ${folderId}.`);
        return downloadedFilePaths;
    }

    // Itera sobre cada elemento (archivo o carpeta)
    for (const item of items) {
        const destPath = path.join(destinationFolder, item.name);

        if (item.mimeType === 'application/vnd.google-apps.folder') {
            // Si es una carpeta de Google Drive, llama recursivamente a la función
            console.log(`Explorando subcarpeta: ${item.name}`);
            const subfolderFiles = await downloadFilesFromDrive(item.id, destPath);
            downloadedFilePaths.push(...subfolderFiles); // Añade los archivos de la subcarpeta al listado general
        } else {
            // Si es un archivo, procede a descargarlo
            console.log(`Descargando archivo: ${item.name} (${item.id}) en ${destinationFolder}`);
            try {
                if (item.mimeType.includes('google-apps')) {
                    // Si es un documento de Google (Docs, Sheets, Slides), exportarlo a texto plano
                    const response = await drive.files.export({
                        fileId: item.id,
                        mimeType: 'text/plain', // Exportar como texto plano
                    }, { responseType: 'stream' });

                    // Escribe el stream de datos en el archivo local
                    await new Promise((resolve, reject) => {
                        response.data
                            .on('end', () => {
                                downloadedFilePaths.push(destPath); // Añade la ruta del archivo descargado
                                resolve();
                            })
                            .on('error', err => {
                                console.error(`Error al exportar y descargar ${item.name}:`, err);
                                reject(err);
                            })
                            .pipe(fs.createWriteStream(destPath));
                    });
                } else {
                    // Si es un archivo normal (PDF, TXT, etc.), descargarlo directamente
                    const response = await drive.files.get({ fileId: item.id, alt: 'media' }, { responseType: 'stream' });

                    // Escribe el stream de datos en el archivo local
                    await new Promise((resolve, reject) => {
                        response.data
                            .on('end', () => {
                                downloadedFilePaths.push(destPath); // Añade la ruta del archivo descargado
                                resolve();
                            })
                            .on('error', err => {
                                console.error(`Error al descargar ${item.name}:`, err);
                                reject(err);
                            })
                            .pipe(fs.createWriteStream(destPath));
                    });
                }
            } catch (error) {
                console.error(`Fallo al descargar ${item.name}:`, error);
            }
        }
    }
    return downloadedFilePaths;
}



/**
 * @function initializeVectorStore
 * @description Carga documentos desde una carpeta local, los divide en chunks
 * y los carga en un Chroma Vector Store remoto.
 * @param {string} docsFolder - La ruta de la carpeta local que contiene los documentos.
 * @returns {Promise<void>} Una promesa que resuelve cuando el vector store ha sido inicializado.
 */
async function initializeVectorStore(docsFolder) {
    try {
        console.log("Inicializando el vector store con Chroma...");
        const collectionName = "documentos_unju";

        const embeddingstext = new GoogleGenerativeAIEmbeddings({
            model: "text-embedding-004", // 768 dimensiones
            taskType: TaskType.RETRIEVAL_DOCUMENT,
            title: "Document title",
        });



        //elimino la collection si ya esta en la bd
        const chromaClient = new ChromaClient({
            //path: "http://10.3.2.199:8000"
            host: "10.3.2.199",
            port: 8000,
            ssl: false, // o true si usas HTTPS
        });
        const colecciones = await chromaClient.listCollections();
        console.log(`Colecciones encontradas: ${colecciones.length}`);
        for (const coleccion of colecciones) {
            const nombre = coleccion.name;
            console.log(`Eliminando colección: ${nombre}`);
            await chromaClient.deleteCollection({ name: nombre });
        }
        console.log("✅ Todas las colecciones fueron eliminadas.");


        // Cargar los documentos y dividirlos en chunks primero
        const loader = new DirectoryLoader(docsFolder, {
            ".txt": (path) => new TextLoader(path),
            ".pdf": (path) => new PDFLoader(path),
        });
        const docs = await loader.load();

        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 5000,
            chunkOverlap: 500,
        });
        const docOutput = await textSplitter.splitDocuments(docs);

        // Arregla metadatos de los documentos para que puedan ser almacenados en chroma
        docOutput.forEach(doc => {
            const cleanMetadata = {};
            for (const key in doc.metadata) {
                const value = doc.metadata[key];
                if (
                    typeof value === "string" ||
                    typeof value === "number" ||
                    typeof value === "boolean" ||
                    value === null
                ) {
                    cleanMetadata[key] = value;
                } else {
                    cleanMetadata[key] = JSON.stringify(value);
                }
            }
            doc.metadata = cleanMetadata;
        }); 

        
        // Usa Chroma.fromDocuments para asegurarte de que la colección
        // se crea con la función de embedding.

  

        vectorStore = await Chroma.fromDocuments(
            docOutput, 
            embeddingstext, 
            {
                collectionName: "documentos_unju",
                url: "http://10.3.2.199:8000",
            }
        );

        //vectorStore = await new Chroma(embeddingstext, {
        //    collectionName: collectionName,
        //    url: "http://10.3.2.199:8000",
            //host: "10.3.2.199",
            //port: 8000,
            //ssl: false, // o true si usas HTTPS
        //});
        //await vectorStore.addDocuments(docOutput);


        console.log("Vector store con Chroma inicializado y documentos cargados con éxito.");

    } catch (error) {
        console.error("Error al inicializar el vector store con Chroma:", error);
    }
}

// Inicia el servidor Express
app.listen(port, async () => {
    console.log(`Server is running on port ${port}`);
    // --- Lógica para DESCARGAR y luego INICIALIZAR el vector store al iniciar el servidor ---
    let downloadedPaths = null; 
    try {
        // -- Logica para descargar archivos de google drive si es necesario
        //const downloadedPaths =  await downloadFilesFromDrive(GOOGLE_DRIVE_FOLDER_ID, LOCAL_DOCS_FOLDER);
        //if (downloadedPaths && downloadedPaths.length > 0) {
            // -- Logica donde elimino carpeta local si corresponde para alvergar a los nuevos archivos 
            //await deleteLocalFolder();
        //}
    } catch (error) {
        console.error("Error crítico durante la descarga de archivos de Google Drive:", error);
    }

    try {
        await initializeVectorStore(LOCAL_DOCS_FOLDER);
        console.log("Proceso de descarga e inicialización completado al inicio del servidor.");
    } catch (error){
        console.error("Error crítico durante la inicialización del vector store al inicio del servidor:", error);
        // Asegura que el vector store esté al menos vacío incluso si hay un error
        vectorStore = new MemoryVectorStore(embeddingstext);
    }

});

/* Endpoint GET para verificar el estado actual del servidor */
app.get('/api/status', async (req, res) => {
    res.json({
        success: true,
        message: 'Server is ok',
    })
})

/* Endpoint GET para hacer preguntas al asistente */
app.get('/api/ask', async (req, res) => {
    let question = "";
    let question_contexto = "";
    try {
        // Verifica si el vector store ha sido inicializado.
        // Si no lo está, devuelve un error 503 (Servicio no disponible).
        if (!vectorStore) {
            return res.status(503).json({ error: 'El vector store no está inicializado. Por favor, espere o verifique los logs del servidor.' });
        }

        const prompt = `Responde primeramente presentandote como Asistente Virtual de la Facultad de Ciencias Económicas de la UNJu, luego agrega una linea en blanco.
                Responde a la pregunta SIEMPRE en el lenguaje español. No obtengas información de otra universidad diferente a la Universidad Nacional de Jujuy.
                Si se pregunta por una materia responder siempre diferenciando a que carrera pertenece la materia. el nombre de una materia puede ser el mismo en diferentes carreras pero son materias diferentes con contenidos minimos diferentes.
                Al final de la respuesta agrega una linea vacia y pon este texto "Si no obtienes resultados puedes plantear la pregunta de otra manera".
                Si la pregunta es en relación a un libro trata de responder en forma de lista de items, puedes buscar en el título o el autor del libro. Cada item es un libro y en cada linea que salga primero el titulo, luego los autores, año de publicacion, eidcion y finalmente la cantidad de ejemplares existentes en biblioteca..
                La lista que este ordenada de forma numerica.
                La pregunta es: `

        if (req.query.pregunta != null && req.query.pregunta !== "") {
            question = req.query.pregunta + " ?";
            question_contexto = prompt + req.query.pregunta + "?";
        } else {
            // Maneja el caso de que no se proporcione una pregunta
            return res.status(400).json({ error: 'Parámetro "pregunta" es requerido y no puede estar vacío.' });
        }

        // Busca el documento más similar en el vector store
        console.log("Pregunta recibida:");
        console.log("Pregunta original:", question);
        console.log("Pregunta con contexto:", question_contexto);

        const resultOne = await vectorStore.similaritySearchWithScore(req.query.pregunta, 5);
        // console.log("Resultados de la búsqueda de similitud:", resultOne);
        console.log(resultOne);

        // Inicializa el modelo de lenguaje
        const llm = new ChatGoogleGenerativeAI({
            model: "gemini-1.5-flash",
            temperature: 0,
            maxRetries: 3,
            apiKey: process.env.GOOGLE_API_KEY,
        });

        // Carga la cadena de preguntas y respuestas
        const chainA = loadQAStuffChain(llm);
        const desde = Date.now(); // Marca de tiempo de inicio
        const resA = await chainA.invoke({
            input_language: "Spanish",
            output_language: "Spanish",
            input_documents: resultOne,
            question: question_contexto,
        });
        const hasta = Date.now(); // Marca de tiempo de fin
        console.log("Segundos transcurridos de la consulta: " + String((hasta - desde) / 1000));
        console.log("Respuesta del LLM:", resA);

        res.json({ result: resA }); // Envía la respuesta como JSON

    } catch (error) {
        console.error("Error en el endpoint /api/ask:", error);
        res.status(500).json({ error: 'Error interno del servidor al procesar la pregunta.' }); // Envía una respuesta de error
    }
})

/* Endpoint GET para recuperar todos los documentos */
app.get('/api/documents/all', async (req, res) => {
    let chromaCollection = null;
    try {
        chromaCollection = vectorStore.collection;
        if (!chromaCollection) {
            return res.status(503).json({ error: 'La colección de Chroma no está inicializada.' });
        }

        // Recupera todos los documentos de la colección
        // La propiedad `include` asegura que obtengas el contenido y los metadatos
        const allDocuments = await chromaCollection.get({
            include: ["metadatas", "documents", "embeddings"]
        });

        const sliceCount = 1;
        res.json({
            success: true,
            documents: {
                ids: allDocuments.ids.slice(0, sliceCount),
                documents: allDocuments.documents.slice(0, sliceCount),
                metadatas: allDocuments.metadatas.slice(0, sliceCount),
                embeddings: allDocuments.embeddings.slice(0, sliceCount)
            }
        });        
    } catch (error) {
        console.error("Error al recuperar todos los documentos:", error);
        res.status(500).json({ error: 'Error interno del servidor al procesar la solicitud.' });
    }
});