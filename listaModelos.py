import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

print("--- MODELOS DISPONIBLES ---")
# Usamos el atributo correcto: supported_actions
for model in client.models.list():
    if 'generateContent' in model.supported_actions:
        # El nombre técnico que necesitas suele estar en model.name
        # Ej: 'models/gemini-1.5-flash'
        model_id = model.name.replace('models/', '')
        print(f"ID para el código: {model_id}")
        print(f"Nombre completo: {model.display_name}")
        print("-" * 30)