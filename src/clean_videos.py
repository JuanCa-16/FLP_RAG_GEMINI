import os
import time
import glob
from dotenv import load_dotenv
from google import genai

# ----------------------------
# CONFIGURACIÓN INICIAL
# ----------------------------
# Cargar la variable GEMINI_API_KEY desde el archivo .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    # Este error se lanza si no encuentra la clave
    raise ValueError("❌ No se encontró GEMINI_API_KEY en el archivo .env. Asegúrate de tenerlo configurado.")

# Inicializar el cliente de la API
client = genai.Client(api_key=api_key)

# Nombres de las carpetas de entrada y salida
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CARPETA_ENTRADA = os.path.join(BASE_DIR, "TXT_METADATA")
CARPETA_SALIDA = os.path.join(BASE_DIR, "TXT_GEMINI")

os.makedirs(CARPETA_SALIDA, exist_ok=True) # Crear la carpeta de salida si no existe

print(f"🌟 Script de limpieza de transcripciones iniciado.")
print(f"   Carpeta de Entrada: {CARPETA_ENTRADA}")
print(f"   Carpeta de Salida: {CARPETA_SALIDA}")
print("-" * 50)


# ----------------------------
# FUNCIONES AUXILIARES
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
        # Si no hay metadata, devuelve todo el contenido como texto
        return contenido.strip(), ""


def limpiar_con_reintentos(texto, intentos=3, espera=10):
    """
    Limpia texto usando Gemini con reintentos para manejar
    errores de sobrecarga o fallos temporales de la API.
    """
    prompt = f"""
    Eres un editor académico especializado en la limpieza y
    reconstrucción de transcripciones de clases universitarias
    provenientes de audio o video.

    OBJETIVO:
    Transformar la transcripción en un texto claro, coherente
    y útil para estudio y para sistemas RAG, preservando el
    contenido académico original.

    REGLAS OBLIGATORIAS:

    1. LIMPIEZA DEL DISCURSO:
    - Eliminar saludos, despedidas y comentarios logísticos.
    - Eliminar muletillas, repeticiones, pausas y ruido del habla.
    - Eliminar frases incompletas o sin valor académico.

    2. REESCRITURA CONTROLADA:
    - Reescribir frases cuando sea necesario para lograr claridad
    y coherencia gramatical.
    - Unificar ideas fragmentadas por el habla oral.
    - No agregar conceptos nuevos ni ejemplos que no estén implícitos
    en la transcripción original.

    3. CONTENIDO ACADÉMICO:
    - Mantener definiciones, explicaciones, razonamientos y pasos técnicos.
    - Conservar el significado académico original del discurso.
    - No simplificar en exceso ni resumir agresivamente.

    4. ESTRUCTURACIÓN PARA RAG:
    - Dividir el texto usando exclusivamente el delimitador ####.
    - Cada fragmento debe ser semánticamente completo y comprensible
    por sí mismo.
    - Cada fragmento debe ser útil para recuperación en un sistema RAG
    sin depender de fragmentos anteriores o posteriores.
    - No fragmentar de forma que se rompan definiciones, ideas centrales
    o razonamientos.

    5. COHERENCIA GLOBAL:
    - Mantener una progresión lógica de los temas.
    - No mezclar conceptos distintos en un mismo fragmento.
    - Priorizar claridad conceptual sobre estilo narrativo.

    RESTRICCIONES FINALES:
    - Salida directa, sin encabezados ni comentarios.
    - No incluir metaexplicaciones ni notas del editor.
    - El resultado debe ser un texto académico claro y utilizable.

    Texto a limpiar:

    {texto}

    """
    # Usamos gemini-2.5-flash: Rápido, eficiente y más económico para tareas de edición de texto.
    MODELO = "gemini-2.5-pro" 

    for i in range(intentos):
        try:
            respuesta = client.models.generate_content(
                model=MODELO,
                contents=prompt
            )
            return respuesta.text
        except Exception as e:
            mensaje = str(e)
            
            # Manejo de errores 429 (sobrecarga) o 503 (servicio no disponible)
            if "UNAVAILABLE" in mensaje or "503" in mensaje or "429" in mensaje:
                print(f"⚠️ API sobrecargada (intento {i+1}/{intentos}). Reintentando en {espera}s...")
                time.sleep(espera)
            # Manejo del error 404 (nombre de modelo incorrecto) o 400 (Bad Request)
            elif "NOT_FOUND" in mensaje or "400" in mensaje:
                print(f"❌ Error fatal de configuración de modelo o clave: {e}")
                return ""
            else:
                print(f"❌ Error no controlado: {e}")
                return ""
                
    print("❌ Falló después de varios intentos. Revisar la conexión y límite de peticiones.")
    return ""


# ----------------------------
# PROCESAMIENTO PRINCIPAL DE ARCHIVOS
# ----------------------------

# Buscar todos los archivos que empiecen por 'video' y terminen en '.txt'
patron_busqueda = os.path.join(CARPETA_ENTRADA, "video*.txt")

archivos_encontrados = glob.glob(patron_busqueda)

if not archivos_encontrados:
    print(f"\n🛑 Error: No se encontraron archivos 'video*.txt' en la carpeta '{CARPETA_ENTRADA}'.")

for ruta_entrada in archivos_encontrados:
    nombre_archivo = os.path.basename(ruta_entrada)
    print(f"🧩 Procesando: {nombre_archivo}")

    try:
        # 1. Leer contenido
        with open(ruta_entrada, "r", encoding="utf-8") as f:
            contenido = f.read()

        # 2. Separar texto y metadata
        texto_original, metadata = separar_metadata(contenido)
        
        if not texto_original:
             print(f"   ⚠️ Archivo vacío o sin texto: {nombre_archivo}. Saltando.")
             continue

        # 3. Limpiar el texto usando Gemini
        texto_limpio = limpiar_con_reintentos(texto_original)

        if not texto_limpio:
            print(f"   ❌ Fallo al limpiar el texto de {nombre_archivo}. Guardando vacío.")
            # Si falla la limpieza, guardamos un marcador de error y la metadata original
            salida = "[ERROR_GEMINI_LIMPIEZA]\n\n" + metadata.strip()
        else:
            # 4. Combinar texto limpio + metadata original
            salida = texto_limpio.strip() + "\n\n" + metadata.strip()

        # 5. Guardar el resultado
        ruta_salida = os.path.join(CARPETA_SALIDA, nombre_archivo)
        with open(ruta_salida, "w", encoding="utf-8") as f:
            f.write(salida)

        print(f"✅ Guardado con éxito: {nombre_archivo}")

    except Exception as e:
        print(f"❌ Error crítico al manejar el archivo {nombre_archivo}: {e}")

print("-" * 50)
print("✨ ¡Proceso de limpieza completado!")