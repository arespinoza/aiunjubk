import cors from 'cors'
import express from 'express'

//import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { GoogleAIFileManager } from "@google/generative-ai/server";

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






// Initialize GoogleGenerativeAI with your API_KEY.
const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
// Initialize GoogleAIFileManager with your API_KEY.
const fileManager = new GoogleAIFileManager(process.env.GOOGLE_API_KEY);
// Upload the file and specify a display name.
const uploadResponse = await fileManager.uploadFile("docs/ingresantes-ayuda.txt", {
  mimeType: "text/plain",
  //displayName: "Gemini 1.5 PDF",
});




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
  chunkSize: 2000,
  chunkOverlap: 200,
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
    try {

      var prompt="Responder en español. "
      if(req.query.pregunta!=null && req.query.pregunta!=""){
        question = prompt + req.query.pregunta;
      };
  
  
      
      // Search for the most similar document
      console.log("Pregunta:");
      console.log(question);

      const resultOne = await vectorStore.similaritySearch(question, 5);
      //console.log("Resultados de la busqueda:");
      //console.log(resultOne);



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
        question,
      });
      var hasta = Date.now();
      console.log("Segundos transcurridos de la consulta: "+String((hasta - desde) / 1000));
      console.log(resA);

      



      const model = genAI.getGenerativeModel({
        // Choose a Gemini model.
        model: "gemini-1.5-flash",
      });
      // Generate content using text and the URI reference for the uploaded file.
      const result = await model.generateContent([
        {
          fileData: {
            mimeType: uploadResponse.file.mimeType,
            fileUri: uploadResponse.file.uri,
          },
        },
        { text: "Puedo ingresar a la facultad si no termine el secundario?" },
      ]);
      console.log("respuesta desde fileupload");
      console.log(result.response.text());
      //resA = result.response.text()

      res.json({ result: resA }); // Send the response as JSON
  
    }  
      catch (error) {
      console.error(error);
      res.status(500).json({ error: 'Internal Server Error' }); // Send an error response
    }
  })

