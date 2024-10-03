import cors from 'cors'
import express from 'express'
import ollama from 'ollama'

import { VertexAI } from "@langchain/google-vertexai-web";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
// This helps connect to our .env file
import * as dotenv from "dotenv";
dotenv.config();

import { OllamaEmbeddings } from "langchain/embeddings/ollama";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai"
import { TaskType } from "@google/generative-ai";

import {TextLoader} from "langchain/document_loaders/fs/text";
import {PDFLoader} from "langchain/document_loaders/fs/pdf";
import {RecursiveCharacterTextSplitter} from "langchain/text_splitter"
import {DirectoryLoader} from "langchain/document_loaders/fs/directory";
import { MemoryVectorStore } from "langchain/vectorstores/memory";


import { Ollama } from "langchain/llms/ollama";
import { loadQAStuffChain} from "langchain/chains";

const app = express();
const port = 3001;
app.use(cors({origin: ['http://localhost:4200','http://10.3.0.62:4200',  'http://aplicaciones.fce.unju.edu.ar', 'http://aplicaciones.unju.edu.ar']}));




const embeddings = new OllamaEmbeddings({
  model: "llama3.2:latest", // default value
  baseUrl: "http://10.3.2.195:11434", // default value
  requestOptions: {
    useMMap: true,
    numThread: 6,
    numGpu: 1,
  },
});


const embeddingstext = new GoogleGenerativeAIEmbeddings({
  model: "text-embedding-004", // 768 dimensions
  taskType: TaskType.RETRIEVAL_DOCUMENT,
  title: "Document title",
});

const embeddingsgoogle = new GoogleGenerativeAIEmbeddings({
  model: "gemini-1.5-flash"
});

//const documents = ["Hello World!", "Bye Bye"];
//const documentEmbeddings = await embeddings.embedDocuments(documents);
//console.log(documentEmbeddings);


// Create docs with a loader
// define what documents to load
const loader = new DirectoryLoader("./docs",{
  ".txt" : (path)=>new TextLoader(path),
  ".pdf" : (path)=>new PDFLoader(path),
})
const docs = await loader.load();
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
})
const docOutput = await textSplitter.splitDocuments(docs)
// Load the docs into the vector store
const vectorStore = await MemoryVectorStore.fromDocuments(
  docOutput,
  embeddingsgoogle
);

console.log(vectorStore);

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
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
    try {
  
      if(req.query.pregunta!=null && req.query.pregunta!=""){
        question = req.query.pregunta;
      };
  
  
      
      // Search for the most similar document
      const resultOne = await vectorStore.similaritySearch(question, 3);
      console.log("Resultados de la busqueda:");
      console.log(resultOne);



      const geminiModel = new ChatGoogleGenerativeAI({
        modelName: "gemini-1.5-flash",
      });
      const llm = new ChatGoogleGenerativeAI({
        model: "gemini-1.5-flash",
        temperature: 0,
        maxRetries: 2,
        // apiKey: "...",
        // other params...
      });

      const llmvertex = new VertexAI({
        model: "gemini-pro",
        temperature: 0,
        maxRetries: 2,
        // other params...
      });      



      const llmA = new Ollama({ 
        model: "llama3.2:latest",
        baseUrl: "http://10.3.2.195:11434", // default value

        });
      const chainA = loadQAStuffChain(geminiModel);
      var desde = Date.now();
      const resA = await chainA.invoke({
        input_documents: resultOne,
        question,
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

  /**
  app.get('/api/ask', async (req, res) => {

    var question = "";
    try {
  
        if(req.query.pregunta!=null && req.query.pregunta!=""){
        question = req.query.pregunta;
        };

        var desde = Date.now();
        const resA = await ollama.chat({
            model: 'gemma:latest',
            messages: [{ role: 'user', content: question}]
        })
        var hasta = Date.now();
        console.log("Segundos transcurridos de la consulta: "+String((hasta - desde) / 1000));
        res.json({ result: resA }); // Send the response as JSON
    }  
      catch (error) {
      console.error(error);
      res.status(500).json({ error: 'Internal Server Error' }); // Send an error response
    }





})

 */