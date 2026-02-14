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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # root

CARPETA_ENTRADA = os.path.join(BASE_DIR, "GIT_FLP")  # Carpeta padre con las clases
CARPETA_SALIDA = os.path.join(BASE_DIR, "GIT_FLP_GEMINI")

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
# EXTRAER URL DEL CONTENIDO
# ----------------------------
def extraer_url(contenido):
    """Extrae la URL que aparece después de 'ORIGEN:' en la primera línea"""
    lineas = contenido.strip().split('\n')
    for linea in lineas[:5]:  # Buscar en las primeras 5 líneas
        if 'ORIGEN:' in linea:
            return linea.replace('ORIGEN:', '').strip()
    return "URL_NO_ENCONTRADA"

# ----------------------------
# CONSTRUIR METADATA
# ----------------------------
def construir_metadata(carpeta_clase, nombre_archivo, url):
    """
    Construye el bloque de metadata en formato estándar.
    Esta función es la ÚNICA responsable de generar metadata.
    """
    return f"""---
METADATA
---
TEMATICA: {carpeta_clase}
FUENTE: GIT
NOMBRE_DOCUMENTO: {nombre_archivo}
URL: {url}"""

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
                print(f"   ⚠️  Reintento {i+1}/{intentos} tras error API...")
                time.sleep(espera)
            else:
                return ""
        except Exception:
            return ""
    return ""

# ----------------------------
# PROMPT PARA ORGANIZAR CONTENIDO WEB
# ----------------------------
def organizar_contenido_web(contenido_original):
    """
    Prompt simplificado: Gemini solo organiza contenido y divide por ###.
    NO genera metadata, NO genera delimitadores adicionales.
    """
    prompt = f"""
Tu ÚNICA tarea es organizar el siguiente contenido educativo en chunks para un sistema RAG.

INSTRUCCIONES:
1. El código puede aparecer con UN CARÁCTER POR LÍNEA - reconstruye el formato correcto
2. Divide el contenido en chunks lógicos usando SOLO el separador ###
3. NO agregues títulos decorativos (como "Definición:", "Características:", etc.)
4. NO agregues ningún tipo de metadata, delimitadores, o información adicional
5. SOLO devuelve el contenido organizado

REGLA CRÍTICA PARA CÓDIGO:
- NUNCA dejes un chunk con SOLO código
- Siempre incluye explicación + código juntos
- Ejemplo correcto:

###
Las funciones en Racket se declaran con lambda. Esta función suma dos números:

(define suma
  (lambda (x y)
    (+ x y)))

FORMATO DE SALIDA (sin nada más):

###
[Chunk 1: texto explicativo y/o código con contexto]

###
[Chunk 2: texto explicativo y/o código con contexto]

###
[Chunk 3: ...]

---
CONTENIDO A ORGANIZAR:
---
{contenido_original}

RECUERDA: 
- Solo organiza y divide por ###
- No agregues metadata ni información extra
- Código siempre con explicación
- Texto limpio y conciso
"""
    return manejar_reintentos(prompt, MODELO)

# ----------------------------
# ENSAMBLAR DOCUMENTO FINAL
# ----------------------------
def ensamblar_documento_final(contenido_organizado, carpeta_clase, nombre_archivo, url):
    """
    Toma el contenido organizado por Gemini y le agrega la metadata al final.
    Esta es la función que construye el documento completo.
    """
    # Limpiar cualquier metadata que Gemini haya intentado agregar
    import re
    contenido_limpio = re.sub(r'---\s*METADATA\s*---[\s\S]*$', '', contenido_organizado).strip()
    
    # Construir metadata usando la función dedicada
    metadata = construir_metadata(carpeta_clase, nombre_archivo, url)
    
    # Ensamblar documento final
    documento_final = f"{contenido_limpio}\n\n{metadata}"
    
    return documento_final

# ----------------------------
# PROCESAMIENTO
# ----------------------------
print("Iniciando procesamiento de archivos web...\n")

archivos_procesados = 0
archivos_con_error = 0

for carpeta_clase in os.listdir(CARPETA_ENTRADA):
    ruta_clase = os.path.join(CARPETA_ENTRADA, carpeta_clase)

    if not os.path.isdir(ruta_clase):
        continue

    print(f"📁 Procesando clase: {carpeta_clase}")

    archivos_txt = glob.glob(os.path.join(ruta_clase, "*.txt"))

    if not archivos_txt:
        print(f"   ⚠️  No se encontraron archivos .txt\n")
        continue

    for ruta_archivo in archivos_txt:
        nombre_archivo = os.path.basename(ruta_archivo)
        print(f"   📄 Procesando: {nombre_archivo}")

        contenido = leer_archivo_normalizado(ruta_archivo)
        if not contenido or not contenido.strip():
            print(f"   ❌ Archivo vacío o no legible\n")
            archivos_con_error += 1
            continue

        # Extraer URL del contenido original
        url = extraer_url(contenido)

        # PASO 1: Gemini organiza el contenido (sin metadata)
        contenido_organizado = organizar_contenido_web(contenido)
        
        # PASO 2: Ensamblar documento final con metadata
        if contenido_organizado and contenido_organizado.strip():
            documento_final = ensamblar_documento_final(
                contenido_organizado, 
                carpeta_clase, 
                nombre_archivo, 
                url
            )
        else:
            # Si Gemini falló, crear documento de error con metadata
            print(f"   ⚠️  Gemini no pudo procesar el contenido")
            contenido_error = f"""###
ERROR: No se pudo procesar el archivo con Gemini

CONTENIDO ORIGINAL:
{contenido[:500]}...
"""
            documento_final = ensamblar_documento_final(
                contenido_error,
                carpeta_clase,
                nombre_archivo,
                url
            )
            archivos_con_error += 1

        # Guardar archivo
        nombre_salida = f"{carpeta_clase}_{os.path.splitext(nombre_archivo)[0]}.txt"
        ruta_salida = os.path.join(CARPETA_SALIDA, nombre_salida)
        
        with open(ruta_salida, "w", encoding="utf-8") as f:
            f.write(documento_final)

        # Verificar metadata
        if "--- METADATA ---" in documento_final or "---\nMETADATA\n---" in documento_final:
            print(f"   ✅ Guardado con metadata: {nombre_salida}")
            archivos_procesados += 1
        else:
            print(f"   ⚠️  ADVERTENCIA: Metadata no encontrada en {nombre_salida}")
            archivos_con_error += 1

        print()

print("=" * 50)
print("✅ Proceso completado exitosamente")
print(f"📂 Archivos guardados en: {CARPETA_SALIDA}")
print(f"\n📊 Estadísticas:")
print(f"   ✅ Procesados correctamente: {archivos_procesados}")
print(f"   ❌ Con errores: {archivos_con_error}")
print(f"   📝 Total: {archivos_procesados + archivos_con_error}")