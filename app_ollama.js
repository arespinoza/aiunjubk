import cors from 'cors'
import express from 'express'
import ollama from 'ollama'

import { OllamaEmbeddings } from "langchain/embeddings/ollama";
import {TextLoader} from "langchain/document_loaders/fs/text";
import {PDFLoader} from "langchain/document_loaders/fs/pdf";
import {RecursiveCharacterTextSplitter} from "langchain/text_splitter"
import {DirectoryLoader} from "langchain/document_loaders/fs/directory";
import { MemoryVectorStore } from "langchain/vectorstores/memory";


import { Ollama } from "langchain/llms/ollama";
import { loadQAStuffChain} from "langchain/chains";

const app = express();
const port = 3000;
app.use(cors({origin: ['http://localhost:4200','http://10.3.0.62:4200',  'http://aplicaciones.fce.unju.edu.ar', 'http://aplicaciones.unju.edu.ar']}));




const embeddings = new OllamaEmbeddings({
  model: "gemma:2b", // default value
  baseUrl: "http://localhost:11434", // default value
  requestOptions: {
    useMMap: true,
    numThread: 6,
    numGpu: 1,
  },
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
      const resultOne = await vectorStore.similaritySearch(question, 5);
      const llmA = new Ollama({ 
        model: "gemma:2b"
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