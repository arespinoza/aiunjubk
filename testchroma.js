import { ChromaClient } from "chromadb";

const CHROMA_URL = "http://10.3.2.199:8000";

const chromaClient = new ChromaClient({
    path: CHROMA_URL,
});


// Nueva función para eliminar colecciones
async function deleteCollections(collectionNames) {
    for (const name of collectionNames) {
        try {
            await chromaClient.deleteCollection({ name });
            console.log(`✅ Colección '${name}' eliminada correctamente.`);
        } catch (error) {
            console.error(`❌ No se pudo eliminar la colección '${name}':`);
            console.error(error.message || error);
        }
    }
}

async function testChromaClient() {
    try {
        const collections = await chromaClient.listCollections();
        console.log("✅ Conexión exitosa con el cliente Chroma. Colecciones disponibles:");
        console.log(collections);
    } catch (error) {
        console.error("❌ No se pudo conectar con Chroma usando el cliente:");
        console.error(error.message || error);
    }
}

async function testHeartbeat() {
    try {
        const response = await fetch(`${CHROMA_URL}/api/v1/heartbeat`);
        const data = await response.json();
        
        if (response.ok) {
            console.log("✅ Conexión HTTP exitosa al endpoint heartbeat:");
            console.log(data);
        } else {
            console.error("❌ Error en la respuesta del servidor para el endpoint heartbeat:", response.status);
        }
    } catch (error) {
        console.error("❌ Falló la conexión HTTP al endpoint heartbeat:");
        console.error(error.message || error);
    }
}

async function queryCollection(searchText) {
    const collectionName = "documentos_unju"; // Reemplaza con el nombre de tu colección
    try {
        const collection = await chromaClient.getCollection({ name: collectionName });
        const results = await collection.query({
            queryTexts: [searchText],
            nResults: 5, // Número de documentos que quieres que te devuelva
        });

        console.log(`✅ Consulta exitosa para el texto: "${searchText}"`);
        console.log("Resultados encontrados:");
        
        // Muestra los documentos encontrados
        if (results.documents && results.documents.length > 0) {
            results.documents[0].forEach((document, index) => {
                console.log(`- Documento ${index + 1}: ${document}`);
            });
        } else {
            console.log("No se encontraron documentos que coincidan.");
        }

    } catch (error) {
        console.error("❌ Falló la consulta a la base de datos:");
        console.error(error);
    }
}

// Ejecuta ambos tests para comparar los resultados
//deleteCollections(["a-test-collection"]);
testChromaClient();
//testHeartbeat();
//queryCollection("¿activo pleno?");

