import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

print("--- MODELOS DISPONIBLES ---")
# Usamos el atributo correcto: supported_actions
for model in client.models.list():
    # Verificamos si sirve para generar texto O para crear embeddings
    if 'generateContent' in model.supported_actions or 'embedContent' in model.supported_actions:
        model_id = model.name.replace('models/', '')
        print(f"ID: {model_id}")
        print(f"Acciones: {model.supported_actions}") # Esto te dirá qué puede hacer cada uno
        print(f"Nombre: {model.display_name}")
        print("-" * 30)