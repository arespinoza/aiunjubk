import cors from 'cors';
import express from 'express';

import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import * as dotenv from "dotenv";
dotenv.config();
console.log("Valor de GOOGLE_API_KEY:", process.env.GOOGLE_API_KEY);

import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";

import { TextLoader } from "langchain/document_loaders/fs/text";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

// --- Importar LanceDB ---
import { LanceDB } from "@langchain/community/vectorstores/lancedb";
import * as lancedb from "@lancedb/lancedb";


import { loadQAStuffChain } from "langchain/chains";

import fs from 'fs';
import path from 'path';

// --- Importaciones y Configuración para Google Drive API ---
import { google } from 'googleapis';
const CLIENT_ID = process.env.GOOGLE_CLIENT_ID;
const CLIENT_SECRET = process.env.GOOGLE_CLIENT_SECRET;
const REDIRECT_URI = process.env.GOOGLE_REDIRECT_URI;
const REFRESH_TOKEN = process.env.GOOGLE_REFRESH_TOKEN;
const oAuth2Client = new google.auth.OAuth2(CLIENT_ID, CLIENT_SECRET, REDIRECT_URI);
oAuth2Client.setCredentials({ refresh_token: REFRESH_TOKEN });
const drive = google.drive({ version: 'v3', auth: oAuth2Client });
// --- Fin de Configuración para Google Drive ---

const app = express();
const port = 3000;
app.use(cors({ origin: ['http://localhost:4200', 'http://10.3.0.62:4200', 'http://aplicaciones.fce.unju.edu.ar', 'http://aplicaciones.unju.edu.ar'] }));

const embeddingstext = new GoogleGenerativeAIEmbeddings({
    model: "text-embedding-004", // 768 dimensions
    taskType: TaskType.RETRIEVAL_DOCUMENT,
    title: "Document title",
});

let vectorStore; // Se declara vectorStore globalmente

// Configuración de LanceDB
const LANCE_DB_PATH = "./lancedb_data"; // Ruta donde se guardará la base de datos LanceDB
const LANCE_DB_TABLE_NAME = "my_documents_table"; // Nombre de la tabla en LanceDB

// Función para descargar archivos desde Google Drive
async function downloadFilesFromDrive(folderId, destinationFolder) {
    if (!fs.existsSync(destinationFolder)) {
        fs.mkdirSync(destinationFolder, { recursive: true });
    }

    const downloadedFilePaths = [];

    const filesList = await drive.files.list({
        q: `'${folderId}' in parents and trashed = false`,
        fields: 'files(id, name, mimeType)',
    });

    const items = filesList.data.files;
    if (items.length === 0) {
        console.log(`No se encontraron archivos ni subcarpetas en ${destinationFolder.split('/').pop()}.`);
        return downloadedFilePaths;
    }

    for (const item of items) {
        const destPath = path.join(destinationFolder, item.name);

        if (item.mimeType === 'application/vnd.google-apps.folder') {
            console.log(`Explorando subcarpeta: ${item.name}`);
            const subfolderFiles = await downloadFilesFromDrive(item.id, destPath);
            downloadedFilePaths.push(...subfolderFiles);
        } else {
            console.log(`Descargando archivo: ${item.name} (${item.id}) en ${destinationFolder}`);
            try {
                if (item.mimeType.includes('google-apps')) {
                    const response = await drive.files.export({
                        fileId: item.id,
                        mimeType: 'text/plain',
                    }, { responseType: 'stream' });

                    await new Promise((resolve, reject) => {
                        response.data
                            .on('end', () => {
                                downloadedFilePaths.push(destPath);
                                resolve();
                            })
                            .on('error', err => {
                                console.error(`Error al descargar ${item.name}:`, err);
                                reject(err);
                            })
                            .pipe(fs.createWriteStream(destPath));
                    });
                } else {
                    const response = await drive.files.get({ fileId: item.id, alt: 'media' }, { responseType: 'stream' });

                    await new Promise((resolve, reject) => {
                        response.data
                            .on('end', () => {
                                downloadedFilePaths.push(destPath);
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

app.listen(port, async () => {
    console.log(`Server is running on port ${port}`);

    const GOOGLE_DRIVE_FOLDER_ID = process.env.GOOGLE_DRIVE_FOLDER_ID;
    const LOCAL_DOCS_FOLDER = './downloaded_docs';

    try {
        // --- LÓGICA PARA ELIMINAR EL CONTENIDO DE LA CARPETA LOCAL ---
        if (fs.existsSync(LOCAL_DOCS_FOLDER)) {
            console.log(`Borrando el contenido de la carpeta local: ${LOCAL_DOCS_FOLDER}`);
            const files = await fs.promises.readdir(LOCAL_DOCS_FOLDER);
            for (const file of files) {
                const filePath = path.join(LOCAL_DOCS_FOLDER, file);
                try {
                    const stats = await fs.promises.lstat(filePath);
                    if (stats.isDirectory()) {
                        await fs.promises.rm(filePath, { recursive: true, force: true });
                        console.log(`   - Subcarpeta eliminada: ${file}`);
                    } else {
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
        // --- FIN LÓGICA ELIMINAR CONTENIDO CARPETA LOCAL ---

        console.log("Descargando archivos desde Google Drive...");
        const downloadedPaths = await downloadFilesFromDrive(GOOGLE_DRIVE_FOLDER_ID, LOCAL_DOCS_FOLDER);

        // --- Configuración e inicialización de LanceDB ---
        const client = await lancedb.connect(LANCE_DB_PATH);
        let table;

        try {
            // Intenta abrir la tabla existente
            table = await client.openTable(LANCE_DB_TABLE_NAME);
            console.log(`Tabla LanceDB '${LANCE_DB_TABLE_NAME}' abierta. Borrando contenido existente...`);
            await client.dropTable(LANCE_DB_TABLE_NAME);
            console.log(`Tabla LanceDB '${LANCE_DB_TABLE_NAME}' eliminada para recrear.`);
            table = undefined; // Aseguramos que se cree de nuevo
        } catch (e) {
            if (e.message.includes("not found")) {
                console.log(`La tabla LanceDB '${LANCE_DB_TABLE_NAME}' no existe. Se creará al cargar documentos.`);
            } else {
                console.error("Error al intentar abrir o limpiar la tabla LanceDB:", e);
                process.exit(1);
            }
        }

        // Si se descargaron documentos, los cargamos y los añadimos a LanceDB
        if (downloadedPaths.length > 0) {
            const loader = new (await import("langchain/document_loaders/fs/directory")).DirectoryLoader(LOCAL_DOCS_FOLDER, {
                ".txt": (path) => new TextLoader(path),
                ".pdf": (path) => new PDFLoader(path),
            });
            const docs = await loader.load();
            const textSplitter = new RecursiveCharacterTextSplitter({
                chunkSize: 5000,
                chunkOverlap: 500,
            });
            let docOutput = await textSplitter.splitDocuments(docs);

            // --- FILTRADO DE METADATOS MÁS AGRESIVO ---
            console.log("Iniciando limpieza de metadatos de documentos...");
            docOutput = docOutput.map(doc => {
                const newMetadata = {}; // Inicia con un objeto de metadatos vacío, solo añade lo que es seguro
                
                for (const key in doc.metadata) {
                    const value = doc.metadata[key];

                    // Excluye explícitamente la propiedad 'pdf' ya que es una fuente común de problemas de PDFLoader
                    if (key === 'pdf') {
                        console.log(`DEBUG: Eliminando metadato 'pdf' del documento.`);
                        continue; // Omite esta propiedad por completo
                    }

                    // Maneja la propiedad 'loc', que también puede ser problemática si su estructura interna es inconsistente
                    if (key === 'loc') {
                        if (value === undefined || value === null || typeof value !== 'object') {
                            console.log(`DEBUG: Eliminando metadato 'loc' problemático (undefined/null/no-objeto).`);
                            continue;
                        }
                        // Si 'loc' es un objeto, intenta serializarlo.
                        // LanceDB espera tipos primitivos o arrays de primitivos para las columnas de metadatos.
                        try {
                            newMetadata[key] = JSON.stringify(value);
                            console.log(`DEBUG: Convertido metadato 'loc' a string.`);
                        } catch (e) {
                            console.warn(`WARN: No se pudo serializar el metadato 'loc'. Eliminando. Error: ${e.message}`);
                            // Si la serialización falla, no lo incluyas
                        }
                        continue; // Pasa a la siguiente clave después de manejar 'loc'
                    }
                    
                    // Para todas las demás claves de metadatos:
                    if (value === undefined || value === null) {
                        console.log(`DEBUG: Eliminando metadato '${key}' con valor undefined/null.`);
                        continue; // No incluyas valores undefined o null
                    } else if (typeof value === 'object' && !Array.isArray(value)) {
                        // Para cualquier otro objeto que no sea un array en los metadatos, conviértelos a string
                        try {
                            newMetadata[key] = JSON.stringify(value);
                            console.log(`DEBUG: Convertido metadato '${key}' a string.`);
                        } catch (e) {
                            console.warn(`WARN: No se pudo serializar el metadato '${key}'. Eliminando. Error: ${e.message}`);
                        }
                    } else {
                        // Incluye valores primitivos (string, number, boolean) y arrays simples
                        newMetadata[key] = value;
                    }
                }
                // console.log("DEBUG: Metadatos finales del documento:", newMetadata); // Descomenta para depuración profunda
                return {
                    ...doc,
                    metadata: newMetadata
                };
            });
            console.log("Limpieza de metadatos completada.");
            // --- FIN DEL FILTRADO DE METADATOS MÁS AGRESIVO ---

            // Carga los documentos en el vector store LanceDB.
            vectorStore = await LanceDB.fromDocuments(
                docOutput,
                embeddingstext,
                { client, tableName: LANCE_DB_TABLE_NAME }
            );
            await vectorStore.table.flush();
            console.log("Vector store inicializado con documentos de Google Drive en LanceDB.");
        } else {
            try {
                table = await client.openTable(LANCE_DB_TABLE_NAME);
                vectorStore = new LanceDB(embeddingstext, { table });
                console.log("No se descargaron documentos. Vector store de LanceDB conectado a la tabla existente.");
            } catch (e) {
                if (e.message.includes("not found")) {
                    console.log(`No se descargaron documentos y la tabla LanceDB '${LANCE_DB_TABLE_NAME}' no existe. Creando tabla con un esquema básico.`);
                    const dummyDoc = {
                        pageContent: "dummy",
                        metadata: {},
                        vector: Array(embeddingstext.model === "text-embedding-004" ? 768 : 0).fill(0)
                    };
                    vectorStore = await LanceDB.fromDocuments(
                        [dummyDoc],
                        embeddingstext,
                        { client, tableName: LANCE_DB_TABLE_NAME }
                    );
                    await vectorStore.delete({ 'pageContent': 'dummy' });
                    console.log("Tabla LanceDB creada y vaciada para futuras adiciones.");
                } else {
                    throw e;
                }
            }
        }
    } catch (error) {
        console.error("Error durante la descarga de archivos de Google Drive o la inicialización del vector store:", error);
        process.exit(1);
    }
});

/* Get endpoint to check current status */
app.get('/api/status', async (req, res) => {
    res.json({
        success: true,
        message: 'Server is ok',
    });
});

app.get('/api/ask', async (req, res) => {
    var question = "";
    var question_contexto = "";
    try {
        var prompt = `Responde primeramente presentandote como Asistente Virtual de la Facultad de Ciencias Económicas de la UNJu, luego agrega una linea en blanco.
                     Responde a la pregunta SIEMPRE en el lenguaje español. No obtengas información de otra universidad diferente a la Universidad Nacional de Jujuy.
                     Si se pregunta por una materia responder siempre diferenciando a que carrera pertenece la materia. el nombre de una materia puede ser el mismo en diferentes carreras pero son materias diferentes con contenidos minimos diferentes.
                     Al final de la respuesta agrega una linea vacia y pon este texto "Si no obtienes resultados puedes plantear la pregunta de otra manera".
                     Si la pregunta es en relación a un libro quiero que sepas que en la fuente de información cada linea representa un libro, se te pide que trates de responder en forma de lista de items, puedes buscar en el título o el autor del libro. Cada item es un libro y en cada linea que salga primero el titulo, luego los autores, año de publicacion, edicion y finalmente la cantidad de ejemplares existentes en biblioteca.
                     Si falta alguno de estos campos en cada libro igual lista lo que encuentres. 
                     La lista que este ordenada de forma numerica.
                     La pregunta es: `;
        if (req.query.pregunta != null && req.query.pregunta != "") {
            question = req.query.pregunta + " ?";
            question_contexto = prompt + req.query.pregunta + "?";
        }

        console.log("Pregunta:");
        console.log(question);
        console.log(question_contexto);

        // --- Búsqueda de similitud en LanceDB ---
        const resultOne = await vectorStore.similaritySearch(question, 20);
        //console.log("Resultados de la busqueda:");
        //console.log(resultOne);

        const llm = new ChatGoogleGenerativeAI({
            model: "gemini-1.5-flash",
            temperature: 0,
            maxRetries: 3,
            apiKey: process.env.GOOGLE_API_KEY,
        });

        const chainA = loadQAStuffChain(llm);
        var desde = Date.now();
        var resA = await chainA.invoke({
            input_language: "Spanish",
            output_language: "Spanish",
            input_documents: resultOne,
            question: question_contexto,
        });
        var hasta = Date.now();
        console.log("Segundos transcurridos de la consulta: " + String((hasta - desde) / 1000));
        console.log(resA);

        res.json({ result: resA }); // Send the response as JSON

    } catch (error) {
        console.error(error);
        res.status(500).json({ error: 'Internal Server Error' }); // Send an error response
    }
});