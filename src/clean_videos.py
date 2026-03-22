import os
import time
import glob
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError

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
CARPETA_SALIDA = os.path.join(BASE_DIR, "NEW_TXT_GEMINI")

os.makedirs(CARPETA_SALIDA, exist_ok=True) # Crear la carpeta de salida si no existe

# ----------------------------
# CONTROL DE RATE LIMITING
# ----------------------------
# Gemini 1.5 Flash: 15 peticiones por minuto
MAX_REQUESTS_PER_MINUTE = 15
REQUEST_INTERVAL = 60.0 / MAX_REQUESTS_PER_MINUTE  # ~4 segundos entre peticiones
last_request_time = 0

def esperar_rate_limit():
    """
    Controla el rate limiting esperando el tiempo necesario entre peticiones.
    """
    global last_request_time
    tiempo_actual = time.time()
    tiempo_transcurrido = tiempo_actual - last_request_time
    
    if tiempo_transcurrido < REQUEST_INTERVAL:
        tiempo_espera = REQUEST_INTERVAL - tiempo_transcurrido
        print(f"   ⏱️  Esperando {tiempo_espera:.1f}s para respetar límite de rate...")
        time.sleep(tiempo_espera)
    
    last_request_time = time.time()


print(f"🌟 Script de limpieza de transcripciones iniciado.")
print(f"   Modelo: Gemini 1.5 Flash")
print(f"   Rate Limit: {MAX_REQUESTS_PER_MINUTE} peticiones/minuto (~{REQUEST_INTERVAL:.1f}s entre peticiones)")
print(f"   Carpeta de Entrada: {CARPETA_ENTRADA}")
print(f"   Carpeta de Salida: {CARPETA_SALIDA}")
print(f"   Filtro: Procesando archivos que empiezan con 'video*.txt'")
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
    prompt = f"""[CONTEXTO]
    Eres un editor técnico especializado en preparar contenido educativo de "Fundamentos de Lenguajes de Programación" para un sistema RAG (Retrieval-Augmented Generation).

    El contenido proviene de: transcripciones de clases, código fuente documentado o material educativo web.

    Tu rol es ser un editor académico que preserva fidelidad absoluta al contenido original mientras lo prepara para recuperación semántica.

    [ACCIÓN]
    Procesa el siguiente texto ejecutando estas tareas en orden:

    1. LIMPIAR: Eliminar ruido del discurso oral (muletillas, saludos, repeticiones sin valor académico)
    2. CORREGIR: Ortografía, gramática, puntuación y sintaxis de código
    3. DIVIDIR: Fragmentar en bloques semánticamente completos usando ####
    4. FORMATEAR: Aplicar sintaxis Markdown correcta a cada fragmento

    Entrega fragmentos separados por #### donde cada uno sea comprensible por sí mismo, esté correctamente formateado en Markdown y preserve el contenido académico original.

    [REGLAS]

    FIDELIDAD ABSOLUTA:
    - NO resumir conceptos
    - NO parafrasear explicaciones técnicas
    - NO cambiar el orden del contenido original
    - NO agregar ejemplos, definiciones o código nuevo
    - SÍ eliminar muletillas sin valor académico ("bueno", "eh", "este")
    - SÍ eliminar saludos, despedidas y referencias logísticas

    CORRECCIÓN TÉCNICA:
    - Corregir ortografía, tildes y puntuación
    - Corregir sintaxis de código, diagramas y figuras si están incorrectos
    - NO cambiar la lógica o funcionalidad del código original
    - Si el código aparece con UN CARÁCTER POR LÍNEA, reconstruir el formato correcto

    CÓDIGO Y DIAGRAMAS:
    - Identificar correctamente el lenguaje o tipo de diagrama
    - Todo código debe estar en sintaxis válida del lenguaje correspondiente
    - Código SIEMPRE debe ir acompañado de explicación en el mismo fragmento
    - NUNCA crear un fragmento que contenga SOLO código sin contexto
    - Usar bloques de código Markdown con especificador correcto (racket, python, mermaid, etc.)

    FRAGMENTACIÓN SEMÁNTICA:
    - Usar EXCLUSIVAMENTE #### como separador entre fragmentos
    - Cada fragmento debe ser autosuficiente y comprensible sin depender de fragmentos anteriores o posteriores
    - No romper definiciones completas, ejemplos o razonamientos
    - No mezclar conceptos distintos en un mismo fragmento
    - Priorizar coherencia temática sobre longitud uniforme

    REESCRITURA CONTROLADA:
    - Unificar ideas fragmentadas por el habla oral en frases coherentes
    - Completar frases incompletas SOLO si el significado es completamente evidente
    - Mantener la terminología técnica exacta del profesor
    - Preservar transiciones académicas naturales ("ahora veremos", "recordemos que")

    PROGRESIÓN LÓGICA:
    - Mantener el orden de presentación de los temas tal como aparecen en la clase
    - No reorganizar contenido por tema si eso rompe el flujo pedagógico original

    FORMATO MARKDOWN:
    - Bloques de código con sintaxis correcta:
    ```lenguaje
    (código aquí)
    ```

    - Usar negrita EXCLUSIVAMENTE para subtítulos de secciones dentro de fragmentos:
    **Subtítulo de sección**

    - Palabras técnicas importantes, conceptos clave y términos resaltados usar comillas invertidas:
    `lambda`, `recursión`, `closure`, `scope léxico`

    - Listas con viñetas para enumeraciones:
    - Punto 1
    - Punto 2

    - NO usar HTML, solo Markdown puro
    - NO agregar títulos decorativos inventados (como "Definición:", "Características:")
    - NO incluir metaexplicaciones del editor

    SALIDA DIRECTA:
    - INICIAR INMEDIATAMENTE con #### y el primer fragmento
    - NO incluir texto previo como "Aquí está tu respuesta:", "He procesado el texto:", "Resultado:"
    - NO incluir encabezados introductorios
    - NO incluir comentarios del editor
    - Solo devolver los fragmentos separados por ####

    CONTENIDO A PROCESAR:
    {texto}

    """
    # Cambiado a gemini-3.1-flash-lite-preview para respetar límite de 15 peticiones/minuto
    MODELO = "gemini-3.1-flash-lite-preview" 

    for i in range(intentos):
        try:
            # Esperar antes de hacer la petición para respetar rate limit
            esperar_rate_limit()
            
            respuesta = client.models.generate_content(
                model=MODELO,
                contents=prompt
            )
            
            # Extraer solo las partes de texto para evitar warnings
            texto_completo = ""
            for parte in respuesta.candidates[0].content.parts:
                if hasattr(parte, 'text'):
                    texto_completo += parte.text
            
            return texto_completo if texto_completo else respuesta.text
            
        except APIError as e:
            mensaje = str(e)
            
            # Manejo de errores 429 (sobrecarga) o 503 (servicio no disponible)
            if "RESOURCE_EXHAUSTED" in mensaje or "UNAVAILABLE" in mensaje or "503" in mensaje or "429" in mensaje:
                # Si es error de rate limit (429), esperar más tiempo
                tiempo_espera = espera * (2 if "429" in mensaje else 1)
                print(f"⚠️ API sobrecargada o límite de rate excedido (intento {i+1}/{intentos}). Reintentando en {tiempo_espera}s...")
                time.sleep(tiempo_espera)
            # Manejo del error 404 (nombre de modelo incorrecto) o 400 (Bad Request)
            elif "NOT_FOUND" in mensaje or "400" in mensaje:
                print(f"❌ Error fatal de configuración de modelo o clave: {e}")
                return ""
            else:
                print(f"❌ Error no controlado: {e}")
                return ""
        except Exception as e:
            print(f"❌ Error no controlado de la librería: {e}")
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
else:
    print(f"\n--- INICIANDO LIMPIEZA ({len(archivos_encontrados)} archivos encontrados) ---")
    tiempo_estimado = len(archivos_encontrados) * REQUEST_INTERVAL
    print(f"⏱️  Tiempo estimado mínimo: {tiempo_estimado/60:.1f} minutos ({REQUEST_INTERVAL:.1f}s por archivo)\n")

total_procesados = 0

for idx, ruta_entrada in enumerate(archivos_encontrados, 1):
    nombre_archivo = os.path.basename(ruta_entrada)
    print(f"\n🧩 [{idx}/{len(archivos_encontrados)}] Procesando: {nombre_archivo}")

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

        print(f"✅ Guardado con éxito en: {os.path.basename(CARPETA_SALIDA)}/{nombre_archivo}")
        total_procesados += 1

    except Exception as e:
        print(f"❌ Error crítico al manejar el archivo {nombre_archivo}: {e}")

print("-" * 50)
print(f"✨ ¡Proceso de limpieza completado! Total de archivos procesados: {total_procesados}")