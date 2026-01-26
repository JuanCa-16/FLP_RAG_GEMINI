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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARPETA_ENTRADA = os.path.join(BASE_DIR, "TXT_METADATA")
CARPETA_SALIDA_ESTRUCTURA = os.path.join(BASE_DIR, "TXT_GEMINI")

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
    prompt = f"""
        Actúa como editor técnico de transcripciones académicas en
        Fundamentos de Lenguajes de Programación y Racket.

        OBJETIVO:
        Normalizar y estructurar la transcripción manteniendo el
        contenido original con la máxima fidelidad posible.

        REGLAS OBLIGATORIAS:

        1. FIDELIDAD AL DISCURSO:
        - No resumir.
        - No parafrasear.
        - No cambiar el orden del contenido.
        - No agregar contenido conceptual nuevo.
        - Eliminar únicamente muletillas orales que no aporten significado técnico.

        2. CORRECCIÓN LINGÜÍSTICA:
        - Corregir ortografía, tildes y puntuación.
        - Mantener el sentido exacto de cada frase.

        3. CÓDIGO (GLOBAL, INCLUYENDO ETIQUETAS):
        - Todo código presente en el texto, sin excepción, debe estar en sintaxis válida de Racket.
        - Esto incluye código dentro de <CODIGO_BLOCK>, ejemplos inline y referencias parciales.
        - Si el código original es incorrecto o incompleto, corregirlo estrictamente
        sin añadir funcionalidades, ejemplos nuevos ni explicaciones extra.

        4. ETIQUETAS ESPECIALES:
        - <IMG_BLOCK>: Reemplazar por una descripción técnica breve únicamente si la imagen
        es necesaria para comprender el contenido; de lo contrario, eliminarla.
        - <CODIGO_BLOCK>: Mantener el código corregido y añadir una explicación técnica mínima,
        directamente relacionada con el discurso original.

        5. FRAGMENTACIÓN OPTIMIZADA PARA RAG:
        - Dividir el documento en fragmentos usando exclusivamente el delimitador ####.
        - Cada fragmento debe ser semánticamente completo y comprensible por sí mismo.
        - Cada fragmento debe conservar el significado original del discurso,
        sin depender de fragmentos anteriores o posteriores.
        - No fragmentar de forma que se rompan definiciones, ejemplos o razonamientos.
        - La fragmentación debe priorizar utilidad para recuperación semántica
        sin alterar el contenido ni el sentido original.

        RESTRICCIONES FINALES:
        - Salida directa, sin encabezados ni comentarios.
        - No interpretar, no opinar, no embellecer el texto.
        - Priorizar fidelidad y corrección técnica por encima de estilo.

        Texto a procesar:

        {texto}

        """
    MODELO = "gemini-2.5-pro" 
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