import cors from 'cors'
import express from 'express'

//import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
// This helps connect to our .env file
import * as dotenv from "dotenv";
dotenv.config();

//import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai"
import { GoogleGenerativeAIEmbeddings }  from "@langchain/google-genai";


import { TaskType } from "@google/generative-ai";

import {TextLoader} from "langchain/document_loaders/fs/text";
import {PDFLoader} from "langchain/document_loaders/fs/pdf";
import {RecursiveCharacterTextSplitter} from "langchain/text_splitter"
//import {DirectoryLoader} from "langchain/document_loaders/fs/directory";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { loadQAStuffChain} from "langchain/chains";



// --- Importaciones y Configuración para Google Drive API ---
import { google } from 'googleapis';
import fs from 'fs';
import path from 'path';
// Configuración de la API de Google Drive
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
app.use(cors({origin: ['http://localhost:4200','http://10.3.0.62:4200',  'http://aplicaciones.fce.unju.edu.ar', 'http://aplicaciones.unju.edu.ar']}));


const embeddingstext = new GoogleGenerativeAIEmbeddings({
  model: "text-embedding-004", // 768 dimensions
  taskType: TaskType.RETRIEVAL_DOCUMENT,
  title: "Document title",
});





// Create docs with a loader
// define what documents to load
////const loader = new DirectoryLoader("./docs",{
////  ".txt" : (path)=>new TextLoader(path),
////  ".pdf" : (path)=>new PDFLoader(path),
////})
////const docs = await loader.load();
////const textSplitter = new RecursiveCharacterTextSplitter({
////  chunkSize: 5000,
////  chunkOverlap: 500,
////})
////const docOutput = await textSplitter.splitDocuments(docs)
// Load the docs into the vector store
////const vectorStore = await MemoryVectorStore.fromDocuments(
////  docOutput,
////  embeddingstext
////);



let vectorStore; // Se declara vectorStore globalmente para que sea accesible desde el endpoint /api/ask
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
            // Si es una carpeta, llama recursivamente a la función
            console.log(`Explorando subcarpeta: ${item.name}`);
            const subfolderFiles = await downloadFilesFromDrive(item.id, destPath);
            downloadedFilePaths.push(...subfolderFiles); // Añade los archivos de la subcarpeta
        } else {
            // Si es un archivo, descárgalo
            console.log(`Descargando archivo: ${item.name} (${item.id}) en ${destinationFolder}`);
            try {
                if (item.mimeType.includes('google-apps')) {
                    const response = await drive.files.export({
                        fileId: item.id,
                        mimeType: 'text/plain', // O 'application/pdf'
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




app.listen(port, async() => {
  console.log(`Server is running on port ${port}`);



    // --- Lógica de integración con Google Drive movida aquí ---
    const GOOGLE_DRIVE_FOLDER_ID = process.env.GOOGLE_DRIVE_FOLDER_ID; // El ID de tu carpeta de Google Drive
    const LOCAL_DOCS_FOLDER = './downloaded_docs'; // Carpeta para guardar los archivos descargados

    try {


        // --- NUEVA LÓGICA PARA ELIMINAR  EL CONTENIDO DE LA CARPETA LOCAL ---
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
                        console.log(`  - Subcarpeta eliminada: ${file}`);
                    } else {
                        // Si es un archivo, lo borra
                        await fs.promises.unlink(filePath);
                        console.log(`  - Archivo eliminado: ${file}`);
                    }
                } catch (err) {
                    console.error(`  - Error al borrar ${filePath}:`, err);
                }
            }
            console.log(`Contenido de la carpeta '${LOCAL_DOCS_FOLDER}' borrado.`);
        } else {
            console.log(`La carpeta '${LOCAL_DOCS_FOLDER}' no existe. Creándola...`);
            fs.mkdirSync(LOCAL_DOCS_FOLDER, { recursive: true });
        }
        // --- FIN NUEVA LÓGICA ---



        console.log("Descargando archivos desde Google Drive...");
        const downloadedPaths = await downloadFilesFromDrive(GOOGLE_DRIVE_FOLDER_ID, LOCAL_DOCS_FOLDER);

        if (downloadedPaths.length > 0) {
            // Ahora, carga los documentos descargados
            const loader = new (await import("langchain/document_loaders/fs/directory")).DirectoryLoader(LOCAL_DOCS_FOLDER, {
                ".txt": (path) => new TextLoader(path),
                ".pdf": (path) => new PDFLoader(path),
            });
            const docs = await loader.load();
            const textSplitter = new RecursiveCharacterTextSplitter({
                chunkSize: 5000,
                chunkOverlap: 500,
            });
            const docOutput = await textSplitter.splitDocuments(docs);
            // Carga los documentos en el vector store
            vectorStore = await MemoryVectorStore.fromDocuments(
                docOutput,
                embeddingstext
            );
            console.log("Vector store inicializado con documentos de Google Drive.");
        } else {
            console.log("No se descargaron documentos de Google Drive. El vector store estará vacío.");
            vectorStore = new MemoryVectorStore(embeddingstext); // Inicializa un vector store vacío
        }
    } catch (error) {
        console.error("Error durante la descarga de archivos de Google Drive o la inicialización del vector store:", error);
        // Inicializa un vector store vacío incluso si hay un error para evitar fallos
        vectorStore = new MemoryVectorStore(embeddingstext);
    }
    // --- Fin de la lógica de integración con Google Drive ---



});

/* Get endpoint to check current status  */
app.get('/api/status', async (req, res) => {
  res.json({
    success: true,
    message: 'Server is ok',
  })
})



app.get('/api/ask', async (req, res) => {
  var question = "";
  var question_contexto = "";
  try {

    var prompt=`Responde primeramente presentandote como Asistente Virtual de la Facultad de Ciencias Económicas de la UNJu, luego agrega una linea en blanco.
                Responde a la  pregunta SIEMPRE en el lenguaje español. No obtengas información de otra universidad diferente a la Universidad Nacional de Jujuy.
                Si se pregunta por una materia responder siempre diferenciando a que carrera pertenece la materia. el nombre de una materia puede ser el mismo en diferentes carreras pero son materias diferentes con contenidos minimos diferentes.
                Al final de la respuesta agrega una linea vacia y pon este texto "Si no obtienes resultados puedes plantear la pregunta de otra manera".
                Si la pregunta es en relación a un libro trata de responder en forma de lista de items, puedes buscar en el título o el autor del libro. Cada item es un libro y en cada linea que salga primero el titulo, luego los autores y finalmente la cantidad de ejemplares existentes en biblioteca.
                La lista que este ordenada de forma numerica.
                La pregunta es: `
    if(req.query.pregunta!=null && req.query.pregunta!=""){
      question = req.query.pregunta +" ?";
      question_contexto = prompt + req.query.pregunta + "?";
    };


    
    // Search for the most similar document
    console.log("Pregunta:");
    console.log(question);
    console.log(question_contexto);

    const resultOne = await vectorStore.similaritySearch(question, 50);
    //console.log("Resultados de la busqueda:");
    //console.log(resultOne);



    const llm = new ChatGoogleGenerativeAI({
      //modelName: "gemini-1.5-pro",
      modelName: "gemini-1.5-flash",
      temperature: 0,
      maxRetries: 3,
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
    console.log("Segundos transcurridos de la consulta: "+String((hasta - desde) / 1000));
    console.log(resA);

    
    res.json({ result: resA }); // Send the response as JSON

  }  
    catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Internal Server Error' }); // Send an error response
  }
})