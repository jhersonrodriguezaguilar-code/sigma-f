# Ejecutar SIGMA-F de forma local

1. Instala Node.js 18 o una versión posterior.
2. Copia `.env.example` como `.env.local`.
3. Reemplaza `pega_tu_clave_aqui` por tu clave de DeepSeek.
4. Abre una terminal en esta carpeta y ejecuta `npm start`.
5. Abre `http://127.0.0.1:4173` en el navegador.

La clave nunca se inserta en `dist/index.html`. El navegador envía la evidencia al servidor local y el servidor local consulta a DeepSeek. Si `.env.local` no existe o no tiene una clave válida, SIGMA-F continúa funcionando con OCR y reglas locales.

El modelo predeterminado es `deepseek-flash`, que admite texto e imágenes. Puedes cambiarlo en `.env.local` con `DEEPSEEK_MODEL`. Las imágenes se envían a DeepSeek solo cuando eliges ejecutar el agente con DeepSeek activado; no se almacenan por este servidor local.
