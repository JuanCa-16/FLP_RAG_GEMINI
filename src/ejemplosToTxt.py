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

RUTA_PADRE = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # root

CARPETA_ENTRADA = os.path.join(BASE_DIR, "EJEMPLOS")
CARPETA_SALIDA = os.path.join(BASE_DIR, "TXT_EJEMPLOS")

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
Genera documentación estructurada para un sistema RAG que debe recuperar ejemplos de código.

CONTEXTO: Este código pertenece al tema "{carpeta_tema}" y será buscado por estudiantes universitarios de la clase de Fundamentos de Lenguaje de Programación. Los ejemplos son en lenguaje Racket.

INSTRUCCIONES:
1. Identifica el PROPÓSITO principal del código
2. Extrae CONCEPTOS CLAVE (palabras técnicas importantes)
3. Describe CASOS DE USO concretos
4. Mantén un balance entre brevedad y riqueza semántica

FORMATO OBLIGATORIO:

######## RESUMEN ########
[Una frase de 10-15 palabras describiendo qué problema resuelve el código]

######## CONCEPTOS ########
[Lista de 3-5 conceptos técnicos separados por comas: ej. "variables libres, scope léxico, lambda calculus, análisis sintáctico"]

######## EXPLICACION ########
[4-6 líneas breves, cada una describiendo:
- Línea 1: Funcionalidad principal
- Línea 2-3: Casos de uso específicos con vocabulario variado
- Línea 4-5: Conceptos o técnicas aplicadas
- Línea 6: Relación con otros temas (si aplica)]

Usa sinónimos y variaciones: "verifica", "determina", "analiza", "evalúa", etc.
Incluye el nombre de funciones clave del código.

######## METADATA ########
NOMBRE_DOCUMENTO: {carpeta_tema}/{nombre_archivo}
FUENTE: CODIGO
lenguaje: identifica correctamente el lenguaje

######## CODIGO ########
Copia el código EXACTO sin modificar nada.
{codigo}

---
IMPORTANTE: 
- NO uses frases repetitivas como "Ejemplo de..."
- SÍ usa verbos variados: implementa, demuestra, ilustra, aplica, resuelve
- Incluye términos técnicos exactos que un estudiante buscaría

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
