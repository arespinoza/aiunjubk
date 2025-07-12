import { LanceDB } from "@langchain/community/vectorstores/lancedb";
import * as lancedb from "@lancedb/lancedb";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";
import * as dotenv from "dotenv";
import fs from "fs";
import path from "path";

// Cargar variables de entorno
dotenv.config();

const LANCE_DB_PATH = path.resolve("./lancedb_data_test");
const LANCE_DB_TABLE_NAME = "test_documents";

// Asegurar que el directorio exista
if (!fs.existsSync(LANCE_DB_PATH)) {
  fs.mkdirSync(LANCE_DB_PATH, { recursive: true });
}

// Paso 1: Crear un documento de ejemplo
const dummyDoc = {
  pageContent: "La Facultad de Ciencias Económicas ofrece carreras de grado y posgrado.",
  metadata: {
    fuente: "documento_prueba",
    categoria: "facultad"
  }
};

// Paso 2: Crear embeddings
const embeddings = new GoogleGenerativeAIEmbeddings({
  model: "text-embedding-004",
  taskType: TaskType.RETRIEVAL_DOCUMENT,
  title: "Documento de prueba",
  apiKey: process.env.GOOGLE_API_KEY
});

// Función principal de test
async function runTest() {
  console.log("📦 Conectando LanceDB en:", LANCE_DB_PATH);
  const client = await lancedb.connect(LANCE_DB_PATH);

  // Eliminar tabla previa si existe
  try {
    await client.dropTable(LANCE_DB_TABLE_NAME);
    console.log("🗑️ Tabla previa eliminada.");
  } catch (err) {
    if (!err.message.includes("not found")) {
      console.error("❌ Error al eliminar tabla:", err);
      return;
    }
  }

  // Crear la tabla con el documento directamente
  await client.createTable(LANCE_DB_TABLE_NAME, [dummyDoc]);
  console.log("📋 Tabla creada correctamente con documento inicial.");

  // Abrir la tabla y pasarla a LangChain
  const table = await client.openTable(LANCE_DB_TABLE_NAME);

  const vectorStore = await LanceDB.fromDocuments(
    [dummyDoc],
    embeddings,
    { table }
  );

  console.log("✅ Documento insertado en LanceDB con embeddings.");

  // Verificar lectura directa desde la tabla
  try {
    const openedTable = await client.openTable(LANCE_DB_TABLE_NAME);
    const results = await openedTable.toArrow().then(b => b.toArray());
    console.log("📄 Registros reales en la tabla:", results);
  } catch (err) {
    console.error("❌ No se pudo leer la tabla directamente:", err);
  }

  // Paso 4: Verificar archivos en disco
  const tablePath = path.join(LANCE_DB_PATH, LANCE_DB_TABLE_NAME);
  if (fs.existsSync(tablePath)) {
    console.log("📁 Archivos encontrados en la tabla:");
    console.log(fs.readdirSync(tablePath));
  } else {
    console.warn("⚠️ No se encontró la tabla en disco.");
  }

  // Paso 5: Buscar con similaridad
  const searchResults = await vectorStore.similaritySearch("carreras de posgrado", 1);
  console.log("🔍 Resultado de búsqueda:", searchResults[0]?.pageContent || "No se encontró nada.");
}

// Ejecutar test
runTest().catch(err => {
  console.error("❌ Error al ejecutar test:", err);
});
