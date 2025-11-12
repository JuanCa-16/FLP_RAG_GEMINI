import os
import time
import glob
import re # Necesario para la comprobación del dígito
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError 

# ----------------------------
# CONFIGURACIÓN INICIAL
# ----------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# Nombres de las carpetas de entrada y salida
CARPETA_ENTRADA = "TXT_METADATA"
CARPETA_SALIDA_ESTRUCTURA = "TXT_GEMINI"

# Crear la carpeta de salida si no existe
os.makedirs(CARPETA_SALIDA_ESTRUCTURA, exist_ok=True)

print(f"🌟 Script de Estructuración de Documentos (limpiar_pdfs.py) iniciado.")
print(f"   Carpeta de Entrada: {CARPETA_ENTRADA}")
print(f"   Carpeta de Salida: {CARPETA_SALIDA_ESTRUCTURA}")
print("   Filtro: Procesando archivos que SÍ empiezan con un dígito (1..., 2..., etc.).")
print("-" * 50)


# ----------------------------
# FUNCIONES AUXILIARES Y MANEJO DE ERRORES
# ----------------------------

def separar_metadata(contenido):
    """
    Separa el texto principal y la metadata (si existe) usando
    la etiqueta "--- METADATA ---" como delimitador.
    """
    partes = contenido.split("--- METADATA ---")
    if len(partes) == 2:
        texto, metadata = partes
        return texto.strip(), "--- METADATA ---\n" + metadata.strip()
    else:
        return contenido.strip(), ""


def manejar_reintentos(prompt, modelo, intentos=3, espera=10):
    """Función genérica para manejar la llamada a la API con reintentos."""
    for i in range(intentos):
        try:
            respuesta = client.models.generate_content(
                model=modelo,
                contents=prompt
            )
            return respuesta.text
        except APIError as e:
            mensaje = str(e)
            if "RESOURCE_EXHAUSTED" in mensaje or "UNAVAILABLE" in mensaje or "503" in mensaje or "429" in mensaje:
                print(f"⚠️ API sobrecargada o no disponible (intento {i+1}/{intentos}). Reintentando en {espera}s...")
                time.sleep(espera)
            elif "NOT_FOUND" in mensaje or "400" in mensaje:
                print(f"❌ Error fatal de configuración de modelo: {e}")
                return ""
            else:
                print(f"❌ Error API no controlado: {e}")
                return ""
        except Exception as e:
            print(f"❌ Error no controlado de la librería: {e}")
            return ""
            
    print("❌ Falló después de varios intentos.")
    return ""

# ----------------------------
# FUNCIÓN DE PROCESAMIENTO GEMINI: ESTRUCTURACIÓN
# ----------------------------

def estructurar_con_reintentos(texto):
    """Estructura el texto en grupos lógicos usando Gemini."""
    prompt = f"""
    Eres un organizador de documentos experto. Tu tarea es analizar el siguiente texto 
    y dividirlo en grupos temáticos o secciones lógicas.

    Debes seguir estas reglas estrictas:
    1. **NO debes reescribir, resumir, eliminar o alterar** el contenido original de ninguna manera. 
       El texto de salida debe ser idéntico al de entrada, excepto por el formato de las secciones.
    2. Identifica los cambios de tema o subtema significativos y usa **'#####'** (cinco almohadillas) 
       en una línea separada para marcar la división entre los grupos lógicos de información.
    3. Asegúrate de que los bloques de código o imagen, delimitados por las etiquetas:
       - **<CODIGO_BLOCK_START>** hasta **<CODIGO_BLOCK_END>**
       - **<IMG_BLOCK_START>** hasta **<IMG_BLOCK_END>**
       se mantengan completamente **íntegros y sin ninguna modificación** en la salida.
    4. El resultado final debe ser el texto original, pero estructurado y dividido de la mejor manera 
       posible en grupos coherentes separados por '#####'.
    5. La unica modificacion posible es en correcciones ortograficas.

    Texto a estructurar:
    ---
    {texto}
    ---
    """
    MODELO = "gemini-2.5-flash" 
    return manejar_reintentos(prompt, MODELO)


# ----------------------------
# PROCESAMIENTO PRINCIPAL CON FILTRO INVERTIDO
# ----------------------------

# Buscar todos los archivos .txt en la carpeta de entrada
patron_busqueda = os.path.join(CARPETA_ENTRADA, "*.txt")
archivos_encontrados = glob.glob(patron_busqueda)

if not archivos_encontrados:
    print(f"\n🛑 Error: No se encontraron archivos '*.txt' en la carpeta '{CARPETA_ENTRADA}'.")
    
total_procesados = 0
archivos_a_estructurar = []

# --- 1. Aplicar el Filtro: Solo deben empezar con un dígito ---
for ruta in archivos_encontrados:
    nombre_archivo = os.path.basename(ruta)
    
    # Comprobación clave: usamos re.match(r'^\d', nombre_archivo)
    # Si EMPIEZA con dígito (r'^\d'), la condición es True y se procesa.
    if re.match(r'^\d', nombre_archivo):
        archivos_a_estructurar.append(ruta)
    else:
        # Si NO empieza con dígito, se salta
        print(f"   ⏩ Saltando: {nombre_archivo} (NO empieza con dígito)")


print(f"\n--- INICIANDO ESTRUCTURACIÓN ({len(archivos_a_estructurar)} archivos filtrados) ---")

# --- 2. Procesamiento de los archivos filtrados ---
for ruta_entrada in archivos_a_estructurar:
    nombre_archivo = os.path.basename(ruta_entrada)
    print(f"\n🧩 Procesando: {nombre_archivo}")

    try:
        # 1. Leer contenido
        with open(ruta_entrada, "r", encoding="utf-8") as f:
            contenido = f.read()

        # 2. Separar texto y metadata
        texto_original, metadata = separar_metadata(contenido)
        
        if not texto_original:
            print(f"   ⚠️ Archivo vacío o sin texto. Saltando.")
            continue

        # 3. Estructurar el texto usando Gemini
        texto_procesado = estructurar_con_reintentos(texto_original)

        if not texto_procesado:
            print(f"   ❌ Fallo al estructurar. Guardando marcador de error.")
            salida = f"[ESTRUCTURACION_ERROR_GEMINI]\n\n" + metadata.strip()
        else:
            # 4. Combinar texto procesado + metadata original
            salida = texto_procesado.strip() + "\n\n" + metadata.strip()

        # 5. Guardar el resultado
        ruta_salida = os.path.join(CARPETA_SALIDA_ESTRUCTURA, nombre_archivo)
        with open(ruta_salida, "w", encoding="utf-8") as f:
            f.write(salida)

        print(f"✅ Guardado con éxito en: {os.path.basename(CARPETA_SALIDA_ESTRUCTURA)}/{nombre_archivo}")
        total_procesados += 1

    except Exception as e:
        print(f"❌ Error crítico al manejar el archivo {nombre_archivo}: {e}")

print("-" * 50)
print(f"✨ ¡Proceso de estructuración completado! Total de archivos estructurados: {total_procesados}")