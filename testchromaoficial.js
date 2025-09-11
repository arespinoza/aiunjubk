import { Chroma } from "@langchain/community/vectorstores/chroma";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";

import * as dotenv from "dotenv";
dotenv.config();

const apiKey = process.env.GOOGLE_API_KEY;
console.log("Valor de GOOGLE_API_KEY:", apiKey);

// Corregido: Pasa la API key explícitamente al constructor
//const embeddings = new GoogleGenerativeAIEmbeddings({
//  apiKey: apiKey,
//});

const embeddings = new GoogleGenerativeAIEmbeddings({
    model: "text-embedding-004", // 768 dimensiones
    taskType: TaskType.RETRIEVAL_DOCUMENT,
    title: "Document title",
});

const vectorStore = await new Chroma(
    embeddings, 
    {
        collectionName: "a-test-collection",
        url: "http://10.3.2.199:8000",
    }
);



// La forma correcta para archivos JavaScript
import { Document } from "@langchain/core/documents"; 

const document1 = {
  pageContent: "The powerhouse of the cell is the mitochondria",
  metadata: { source: "https://example.com" }
};

const document2 = {
  pageContent: "Buildings are made out of brick",
  metadata: { source: "https://example.com" }
};

const document3 = {
  pageContent: "Mitochondria are made out of lipids",
  metadata: { source: "https://example.com" }
};

const document4 = {
  pageContent: "The 2024 Olympics are in Paris",
  metadata: { source: "https://example.com" }
}

const documents = [document1, document2, document3, document4];

await vectorStore.addDocuments(documents, { ids: ["1", "2", "3", "4"] });

// --- Verificamos los datos realizando una búsqueda de similitud ---
const query = "¿powerhouse of the cell?";
console.log(`\nBuscando documentos relacionados con: "${query}"`);

const retrievedDocs = await vectorStore.similaritySearch(query);

console.log("\nDocumentos encontrados:");
console.log(retrievedDocs);