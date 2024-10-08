import { MemoryVectorStore } from "langchain/vectorstores/memory";

import { OpenAIEmbeddings } from "@langchain/openai";
import { OpenAI } from "langchain/llms/openai";

import {DirectoryLoader} from "langchain/document_loaders/fs/directory";
import {TextLoader} from "langchain/document_loaders/fs/text";
import {PDFLoader} from "langchain/document_loaders/fs/pdf";
import {RecursiveCharacterTextSplitter} from "langchain/text_splitter"

import { LlamaCpp } from "langchain/llms/llama_cpp";
import { loadQAStuffChain} from "langchain/chains";

//import { LlamaCppEmbeddings } from "@langchain/community/embeddings/llama_cpp";
import { LlamaCppEmbeddings } from "langchain/embeddings/llama_cpp";

import * as dotenv from 'dotenv'
dotenv.config();

import cors from 'cors'
import express from 'express'
import http from 'http'
import { time } from "console";


const app = express();
const port = 3000;
app.use(cors({origin: ['http://localhost:4200','http://10.3.0.62:4200',  'http://aplicaciones.fce.unju.edu.ar', 'http://aplicaciones.unju.edu.ar']}));


/* Create HTTP server */
http.createServer(app).listen(process.env.PORT)
console.info('listening on port ' + process.env.PORT)





const embeddings = new LlamaCppEmbeddings(
  {
    modelPath: "./models/llama-3-neural-chat-v1-8b-Q4_K_M.gguf",
  }
);
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
  embeddings
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
  try {

    if(req.query.pregunta!=null && req.query.pregunta!=""){
      question = req.query.pregunta;
    };


    
    // Search for the most similar document
    const resultOne = await vectorStore.similaritySearch(question, 3);
    const llmA = new LlamaCpp({ 
        modelPath: "./models/llama-3-neural-chat-v1-8b-Q4_K_M.gguf",
        lang_code: "es"
      });
    const chainA = loadQAStuffChain(llmA);
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
app.get('/api/askopenai', async (req, res) => {
  var question = "";
  try {

    if(req.query.pregunta!=null && req.query.pregunta!=""){
      question = req.query.pregunta;
    };


    
    const llmA = new OpenAI({ modelName: "gpt-3.5-turbo"});
    const chainA = loadQAStuffChain(llmA);
    const directory = './docs' //saved directory in .env file
    
    //const loadedVectorStore = await FaissStore.load(
    //  directory,
    //  new OpenAIEmbeddings()
    //  );
      
    const result = await vectorStore.similaritySearch(question, 3);
    const resA = await chainA.call({
      input_documents: result,
      question,
    });

    res.json({ result: resA}); // Send the response as JSON

  }  
    catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Internal Server Error' }); // Send an error response
  }
})
 */