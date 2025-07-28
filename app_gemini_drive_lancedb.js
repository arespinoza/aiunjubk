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
import { loadQAStuffChain } from "langchain/chains";
import { DirectoryLoader }  from "langchain/document_loaders/fs/directory";

import fs from 'fs';
import path from 'path';

// --- Importaciones LanceDB ---
import { LanceDB } from "@langchain/community/vectorstores/lancedb";
// ¡IMPORTANTE! Se importa el paquete completo y luego se desestructuran sus exports.
import pkg from "@lancedb/lancedb";
const { connect, FixedSizeList, Float32, Utf8, Struct } = pkg;

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

let vectorStore;
let lancedbClient;
let lancedbTable;

const GOOGLE_DRIVE_FOLDER_ID = process.env.GOOGLE_DRIVE_FOLDER_ID;
const LOCAL_DOCS_FOLDER = './downloaded_docs';

const LANCE_DB_PATH = path.resolve("./lancedb_data"); // Ruta para LanceDB
const LANCE_DB_TABLE_NAME = "fce_unju_documents"; // Nombre de la tabla LanceDB

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
    if (!items || items.length === 0) {
        console.log(`No se encontraron archivos ni subcarpetas en la carpeta de Google Drive con ID: ${folderId}.`);
        return downloadedFilePaths;
    }

    // Itera sobre cada elemento (archivo o carpeta)
    for (const item of items) {
        if (!item.name || !item.id || !item.mimeType) {
            console.warn(`Elemento incompleto encontrado, omitiendo: ${JSON.stringify(item)}`);
            continue;
        }
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
 * y los carga en LanceDB.
 * @param {string} docsFolder - La ruta de la carpeta local que contiene los documentos.
 * @returns {Promise<void>} Una promesa que resuelve cuando el vector store ha sido inicializado.
 */
async function initializeVectorStore(docsFolder) {
    try {
        console.log("Inicializando el vector store con documentos de la carpeta local en LanceDB...");

        // Conectar a LanceDB usando 'connect' directamente
        lancedbClient = await connect(LANCE_DB_PATH);
        console.log(`Conectado a LanceDB en: ${LANCE_DB_PATH}`);

        // 1. Intentar borrar la tabla si existe para una inicialización limpia.
        try {
            await lancedbClient.dropTable(LANCE_DB_TABLE_NAME);
            console.log(`Tabla '${LANCE_DB_TABLE_NAME}' eliminada para una inicialización limpia.`);
        } catch (err) {
            if (err.message && !err.message.includes("not found")) {
                console.warn(`Advertencia al intentar eliminar tabla (no fue por "not found"): ${err.message}`);
            } else {
                console.log(`La tabla '${LANCE_DB_TABLE_NAME}' no existe, se creará una nueva.`);
            }
        }

        // 2. Cargar los documentos y dividirlos en chunks.
        const loader = new DirectoryLoader(docsFolder, {
            ".txt": (path) => new TextLoader(path),
            ".pdf": (path) => new PDFLoader(path),
        });
        const docs = await loader.load();
        
        let docOutput = [];
        if (docs.length > 0) {
            const textSplitter = new RecursiveCharacterTextSplitter({
                chunkSize: 5000,
                chunkOverlap: 500,
            });
            docOutput = await textSplitter.splitDocuments(docs);
            console.log(`Se han dividido los documentos en ${docOutput.length} chunks.`);

            // --- VALIDACIÓN DE CHUNKS ---
            const initialDocOutputCount = docOutput.length;
            docOutput = docOutput.filter(doc => 
                doc.pageContent && typeof doc.pageContent === 'string' && doc.pageContent.trim().length > 0
            );
            if (docOutput.length < initialDocOutputCount) {
                console.warn(`Advertencia: Se filtraron ${initialDocOutputCount - docOutput.length} chunks con pageContent vacío o inválido.`);
            }
            // --- FIN VALIDACIÓN ---

        } else {
            console.log("No se encontraron documentos para cargar.");
        }

        // 3. Obtener la dimensión del vector desde el modelo de embeddings.
        const vectorDimension = 768; 

        // 4. Crear la tabla explícitamente con un esquema robusto y la dimensión del vector.
        lancedbTable = await lancedbClient.createTable(LANCE_DB_TABLE_NAME, [
            {
                name: "vector", 
                type: FixedSizeList(Float32, vectorDimension), // Usa FixedSizeList y Float32 directamente
                nullable: false, 
            },
            { name: "pageContent", type: Utf8, nullable: false }, // Usa Utf8 directamente
            { name: "metadata", type: Struct(), nullable: false } // Usa Struct directamente
        ]);
        console.log(`Tabla '${LANCE_DB_TABLE_NAME}' creada o abierta con éxito.`);

        // 5. Inicializar el vector store de LangChain con la tabla explícitamente abierta.
        vectorStore = new LanceDB(embeddingstext, { table: lancedbTable });

        // 6. Agregar documentos si hay.
        if (docOutput.length > 0) {
            console.log(`Agregando ${docOutput.length} chunks a la tabla LanceDB.`);
            await vectorStore.addDocuments(docOutput);
            console.log("Documentos agregados a LanceDB con éxito.");
        } else {
            console.log("No hay chunks para agregar al vector store. Permanecerá vacío.");
        }

        console.log("Vector store inicializado con éxito en LanceDB.");
    } catch (error) {
        console.error("Error al inicializar el vector store en LanceDB:", error);
        // Lógica de recuperación si la inicialización principal falla
        try {
            // Conectar a LanceDB en modo de recuperación usando 'connect' directamente
            if (!lancedbClient) {
                 lancedbClient = await connect(LANCE_DB_PATH);
            }

            const vectorDimension = 768; // Dimensión para el modo de recuperación también

            // Intenta crear una tabla de recuperación simple y vacía.
            try {
                 lancedbTable = await lancedbClient.createTable(LANCE_DB_TABLE_NAME, [
                    {
                        name: "vector",
                        type: FixedSizeList(Float32, vectorDimension),
                        nullable: false,
                    },
                    { name: "pageContent", type: Utf8, nullable: false },
                    { name: "metadata", type: Struct(), nullable: false }
                ]);
            } catch (createErr) {
                if (createErr.message && createErr.message.includes("already exists")) {
                    console.warn(`Advertencia: La tabla '${LANCE_DB_TABLE_NAME}' ya existía en el modo de recuperación. Abriéndola.`);
                    lancedbTable = await lancedbClient.openTable(LANCE_DB_TABLE_NAME);
                } else {
                    throw createErr; // Re-lanzar si es otro tipo de error de creación
                }
            }
           
            vectorStore = new LanceDB(embeddingstext, { table: lancedbTable });
            console.log("Vector store inicializado en modo de recuperación (vacío).");
        } catch (reconnectError) {
            console.error("Error crítico al intentar inicializar LanceDB en modo de recuperación:", reconnectError);
            vectorStore = null; 
        }
    }
}

// Inicia el servidor Express
app.listen(port, async () => {
    console.log(`Server is running on port ${port}`);

    // Asegurar que el directorio LanceDB exista
    if (!fs.existsSync(LANCE_DB_PATH)) {
        fs.mkdirSync(LANCE_DB_PATH, { recursive: true });
        console.log(`Directorio LanceDB creado en: ${LANCE_DB_PATH}`);
    }

    // --- Lógica para ELIMINAR EL CONTENIDO de la carpeta local antes de descargar ---
    if (fs.existsSync(LOCAL_DOCS_FOLDER)) {
        console.log(`Borrando el contenido de la carpeta local: ${LOCAL_DOCS_FOLDER}`);
        const files = await fs.promises.readdir(LOCAL_DOCS_FOLDER);
        for (const file of files) {
            const filePath = path.join(LOCAL_DOCS_FOLDER, file);
            try {
                const stats = await fs.promises.lstat(filePath);
                if (stats.isDirectory()) {
                    await fs.promises.rm(filePath, { recursive: true, force: true });
                    console.log(`   - Subcarpeta eliminada: ${file}`);
                } else {
                    await fs.promises.unlink(filePath);
                    console.log(`   - Archivo eliminado: ${file}`);
                }
            } catch (err) {
                console.error(`   - Error al borrar ${filePath}:`, err);
            }
        }
        console.log(`Contenido de la carpeta '${LOCAL_DOCS_FOLDER}' borrado.`);
    } else {
        console.log(`La carpeta '${LOCAL_DOCS_FOLDER}' no existe. Creándola...`);
        fs.mkdirSync(LOCAL_DOCS_FOLDER, { recursive: true });
    }
    // --- FIN LÓGICA ELIMINAR CONTENIDO CARPETA LOCAL---

    // --- Lógica para DESCARGAR y luego INICIALIZAR el vector store al iniciar el servidor ---
    try {
        console.log("Iniciando descarga de archivos desde Google Drive...");
        const downloadedPaths = await downloadFilesFromDrive(GOOGLE_DRIVE_FOLDER_ID, LOCAL_DOCS_FOLDER);

        await initializeVectorStore(LOCAL_DOCS_FOLDER); // Siempre intenta inicializar el vector store

        if (downloadedPaths.length > 0) {
            console.log("Proceso de descarga e inicialización completado al inicio del servidor.");
        } else {
            console.log("No se descargaron documentos de Google Drive. El vector store se inicializó (posiblemente vacío).");
        }
    } catch (error) {
        console.error("Error crítico durante la descarga de archivos de Google Drive o la inicialización del vector store al inicio del servidor:", error);
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
            return res.status(400).json({ error: 'Parámetro "pregunta" es requerido y no puede estar vacío.' });
        }

        console.log("Pregunta recibida:");
        console.log("Pregunta original:", question);
        console.log("Pregunta con contexto:", question_contexto);

        const resultOne = await vectorStore.similaritySearch(question, 20);

        const llm = new ChatGoogleGenerativeAI({
            model: "gemini-1.5-flash",
            temperature: 0,
            maxRetries: 3,
            apiKey: process.env.GOOGLE_API_KEY,
        });

        const chainA = loadQAStuffChain(llm);
        const desde = Date.now();
        const resA = await chainA.invoke({
            input_language: "Spanish",
            output_language: "Spanish",
            input_documents: resultOne,
            question: question_contexto,
        });
        const hasta = Date.now();
        console.log("Segundos transcurridos de la consulta: " + String((hasta - desde) / 1000));
        console.log("Respuesta del LLM:", resA);

        res.json({ result: resA });

    } catch (error) {
        console.error("Error en el endpoint /api/ask:", error);
        res.status(500).json({ error: 'Error interno del servidor al procesar la pregunta.' });
    }
})