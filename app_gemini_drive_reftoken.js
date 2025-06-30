import { google } from 'googleapis';
import * as dotenv from "dotenv";
dotenv.config();
import readline from 'readline'; // Importa readline

const CLIENT_ID = process.env.GOOGLE_CLIENT_ID;
const CLIENT_SECRET = process.env.GOOGLE_CLIENT_SECRET;
// Para aplicaciones de escritorio, esta es una URI de redirección común
const REDIRECT_URI = 'https://aplicaciones.fce.unju.edu.ar'; 

const oAuth2Client = new google.auth.OAuth2(CLIENT_ID, CLIENT_SECRET, REDIRECT_URI);

const authUrl = oAuth2Client.generateAuthUrl({
    access_type: 'offline', // Necesario para obtener un refresh token
    scope: ['https://www.googleapis.com/auth/drive.readonly'], // O 'https://www.googleapis.com/auth/drive' si necesitas escribir
});

console.log('Autoriza esta aplicación visitando esta URL:', authUrl);

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
});

rl.question('Introduce el código de la página aquí: ', async (code) => {
    try {
        const { tokens } = await oAuth2Client.getToken(code);
        console.log('Tu Refresh Token es:', tokens.refresh_token);
        console.log('Guarda este token en tu archivo .env');
    } catch (error) {
        console.error('Error al obtener tokens:', error);
    } finally {
        rl.close();
    }
});