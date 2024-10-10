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
import {DirectoryLoader} from "langchain/document_loaders/fs/directory";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { loadQAStuffChain} from "langchain/chains";

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
const loader = new DirectoryLoader("./docs",{
  ".txt" : (path)=>new TextLoader(path),
  ".pdf" : (path)=>new PDFLoader(path),
})
const docs = await loader.load();
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 3000,
  chunkOverlap: 300,
})
const docOutput = await textSplitter.splitDocuments(docs)
// Load the docs into the vector store
const vectorStore = await MemoryVectorStore.fromDocuments(
  docOutput,
  embeddingstext
);

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
    var question_contexto = "";
    try {

      var prompt=`Responde primeramente presentandote como Asistente Virtual de la Facultad de Ciencias Económicas - UNJu.
                  Responde a la  pregunta SIEMPRE en el lenguaje español.
                  Responde poniendote en el contexto de la Facultad de Ciencias Económicas de la Universidad Nacional de Jujuy. `
      if(req.query.pregunta!=null && req.query.pregunta!=""){
        question = req.query.pregunta +" ?";
        question_contexto = prompt + req.query.pregunta;
      };
  
  
      
      // Search for the most similar document
      console.log("Pregunta:");
      console.log(question);

      const resultOne = await vectorStore.similaritySearch(question, 5);
      console.log("Resultados de la busqueda:");
      console.log(resultOne);



      const llm = new ChatGoogleGenerativeAI({
        modelName: "gemini-1.5-pro",
        temperature: 0,
        maxRetries: 2,
      });


      const chainA = loadQAStuffChain(llm);
      var desde = Date.now();
      var resA = await chainA.invoke({
        input_language: "Spanish",
        output_language: "Spanish",
        input_documents: resultOne,
        question_contexto,
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