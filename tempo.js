import 'dotenv/config';
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { Document } from "@langchain/core/documents";
import { TaskType } from "@google/generative-ai";
import { ChromaClient } from "chromadb";

// 1. Inicializar el modelo de embeddings
console.log(process.env.GOOGLE_API_KEY);

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




// 2. Documentos de ejemplo
const docs = [
  new Document({ pageContent: "El Quijote es una novela escrita por Cervantes." }),
  new Document({ pageContent: "La inteligencia artificial está transformando el mundo." }),
  new Document({ pageContent: "LangChain permite crear aplicaciones LLM fácilmente." }),
];


// 3. Crear y poblar la base de datos vectorial Chroma
const vectorStore = await Chroma.fromDocuments(docs, embeddingstext,             {
                collectionName: "documentos_prueba",
                url: "http://10.3.2.199:8000",
});

// 4. Realizar una búsqueda semántica
//const resultados = await vectorStore.similaritySearch("¿Qué es LangChain?", 2);

//console.log("Resultados de la búsqueda:");
//for (const result of resultados) {
//  console.log("-", result.pageContent);
//}
