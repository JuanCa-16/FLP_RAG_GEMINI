import os
import glob
import time
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError

# ----------------------------
# CONFIGURACIÓN INICIAL
# ----------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("No se encontró GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

CARPETA_ENTRADA = "EJEMPLOS"
CARPETA_SALIDA = "TXT_EJEMPLOS"
os.makedirs(CARPETA_SALIDA, exist_ok=True)

MODELO = "gemini-2.5-flash"

# ----------------------------
# LECTURA SEGURA
# ----------------------------
def leer_archivo_normalizado(path):
    for enc in ("utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            pass
    return None

# ----------------------------
# REINTENTOS
# ----------------------------
def manejar_reintentos(prompt, modelo, intentos=3, espera=10):
    for i in range(intentos):
        try:
            resp = client.models.generate_content(
                model=modelo,
                contents=prompt
            )
            return resp.text
        except APIError as e:
            msg = str(e)
            if any(x in msg for x in ["RESOURCE_EXHAUSTED", "UNAVAILABLE", "503", "429"]):
                time.sleep(espera)
            else:
                return ""
        except Exception:
            return ""
    return ""

# ----------------------------
# PROMPT DEFINITIVO
# ----------------------------
def generar_documentacion(carpeta_tema, nombre_archivo, codigo):
    prompt = f"""
Genera documentación para RECUPERACIÓN POR EJEMPLOS (RAG).

PROHIBIDO:
- Párrafos largos
- Explicaciones detalladas
- Teoría
- Narrativa

OBLIGATORIO:
- Frases MUY CORTAS
- Uso de la palabra "Ejemplo" al inicio
- Texto mínimo y directo

FORMATO ESTRICTO (NO CAMBIAR):

######## EXPLICACION ########
Línea 1: UNA sola frase breve que diga qué hace el código como ejemplo.
Líneas siguientes: frases cortas que empiecen con "Ejemplo de ...".
Máximo 6 líneas en total.

######## METADATA ########
NOMBRE_DOCUMENTO: {carpeta_tema}/{nombre_archivo}
FUENTE: CODIGO
lenguaje: identifica correctamente el lenguaje

######## CODIGO ########
Copia el código EXACTO sin modificar nada.

Código:
---
{codigo}
---
"""
    return manejar_reintentos(prompt, MODELO)

# ----------------------------
# PROCESAMIENTO
# ----------------------------
for carpeta_tema in os.listdir(CARPETA_ENTRADA):
    ruta_tema = os.path.join(CARPETA_ENTRADA, carpeta_tema)

    if not os.path.isdir(ruta_tema):
        continue

    archivos = (
        glob.glob(os.path.join(ruta_tema, "*.rkt")) +
        glob.glob(os.path.join(ruta_tema, "*.scm"))
    )

    for ruta_archivo in archivos:
        nombre_archivo = os.path.basename(ruta_archivo)
        print(f"Procesando: {carpeta_tema}/{nombre_archivo}")

        codigo = leer_archivo_normalizado(ruta_archivo)
        if not codigo or not codigo.strip():
            continue

        salida = generar_documentacion(carpeta_tema, nombre_archivo, codigo)
        if not salida:
            salida = "[ERROR_GEMINI]"

        nombre_salida = f"{carpeta_tema}_{os.path.splitext(nombre_archivo)[0]}.txt"
        with open(os.path.join(CARPETA_SALIDA, nombre_salida), "w", encoding="utf-8") as f:
            f.write(salida)

        print(f"✅ Guardado: {nombre_salida}")

print("Proceso terminado.")
