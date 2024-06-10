import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { Chroma} from "langchain/vectorstores/chroma"
import { OpenAIEmbeddings } from "@langchain/openai";
import { OpenAI } from "langchain/llms/openai";

import {DirectoryLoader} from "langchain/document_loaders/fs/directory";
import {TextLoader} from "langchain/document_loaders/fs/text";
import {PDFLoader} from "langchain/document_loaders/fs/pdf";
import {RecursiveCharacterTextSplitter} from "langchain/text_splitter"

import { LlamaCpp } from "langchain/llms/llama_cpp";
import { RetrievalQAChain, loadQAStuffChain} from "langchain/chains";

//import { LlamaCppEmbeddings } from "@langchain/community/embeddings/llama_cpp";
import { LlamaCppEmbeddings } from "langchain/embeddings/llama_cpp";

import * as dotenv from 'dotenv'
dotenv.config();


import { ChromaClient } from "chromadb";
const chroma = new ChromaClient({ path: "http://10.3.2.199:8000" });
const collectionToDelete = await chroma.getOrCreateCollection({ name: "dbfce"});
await chroma.deleteCollection(collectionToDelete);
const collection = await chroma.getOrCreateCollection({ name: "dbfce"});


import cors from 'cors'
import express from 'express'
import http from 'http'


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
    const vectorStore = await Chroma.fromDocuments(
      docOutput,
      embeddings,
      {
        collectionName: "dbfce",
        url: "http://10.3.2.199:8000",
        collectionMetadata: {
          "hnsw:space": "cosine"
        }
      }
    );

    const collectiondbfce = await chroma.getOrCreateCollection({ name: "dbfce"});
    console.log("*************** collection ****************");
    console.log(collectiondbfce);
    console.log("*************** vector ****************");
    console.log(vectorStore)
    const resultOne = await collection.query({
      queryTexts:"pasivo",
      nResults:2,
      queryEmbeddings: embeddings,
      
    })
    console.log("*************** resultOne ****************");
    console.log(resultOne);


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
    
    //defino el objeto embeddings
    const embeddings = new LlamaCppEmbeddings(
      {
        modelPath: "./models/llama-3-neural-chat-v1-8b-Q4_K_M.gguf",
      }
    );

    // Load the docs into the vector store
    const vectorStore = await Chroma.fromExistingCollection(embeddings,{
      collectionName: "dbfce",
      url: "http://10.3.2.199:8000"
    })
    const llmA = new LlamaCpp({ 
      modelPath: "./models/llama-3-neural-chat-v1-8b-Q4_K_M.gguf",
      lang_code: "es"
    });
    const chain = new RetrievalQAChain({
      combineDocumentsChain: loadQAStuffChain(llmA),
      retriever: vectorStore.asRetriever(),
      returnSourceDocuments: true
    })

    const res = await chain.call({
      query: question
    });



    //const collection = await chroma.getOrCreateCollection({ name: "dbfce" });
    //const resultOne = await vectorStore.similaritySearch(question, 3);
    
    /**

    const chainA = loadQAStuffChain(llmA);
    
    const resA = await chainA.call({
      input_documents: resultOne,
      question,
    });
    */
    res.json({ result: resultOne }); // Send the response as JSON

  }  
    catch (error) {
      console.error(error);
      res.status(500).json({ error: 'Internal Server Error' }); // Send an error response
  }
})
