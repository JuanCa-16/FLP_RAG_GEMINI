import os
import time
import glob
import re
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError 


# CONFIGURACIÓN INICIAL
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CARPETA_ENTRADA = os.path.join(BASE_DIR, "data", "txt", "raw", "NEW")
CARPETA_SALIDA_ESTRUCTURA = os.path.join(BASE_DIR,  "data", "txt", "processed", "GEMINI_3_FLASH","GEMINI_PDFS_VIDEOS")
os.makedirs(CARPETA_SALIDA_ESTRUCTURA, exist_ok=True)


# CONTROL DE RATE LIMITING
MODELO = "gemini-3-flash-preview" 
MAX_REQUESTS_PER_MINUTE = 5
REQUEST_INTERVAL = 60.0 / MAX_REQUESTS_PER_MINUTE  
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


print(f"🌟 Script de Estructuración de Documentos (limpiar_pdfs.py) iniciado.")
print(f"   Modelo: : {MODELO} ")
print(f"   Rate Limit: {MAX_REQUESTS_PER_MINUTE} peticiones/minuto (~{REQUEST_INTERVAL:.1f}s entre peticiones)")
print(f"   Carpeta de Entrada: {CARPETA_ENTRADA}")
print(f"   Carpeta de Salida: {CARPETA_SALIDA_ESTRUCTURA}")
print("   Filtro: Procesando archivos que SÍ empiezan con un dígito (1..., 2..., etc.).")
print("-" * 50)



# FUNCIONES AUXILIARES Y MANEJO DE ERRORES
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
            # Esperar antes de hacer la petición para respetar rate limit
            esperar_rate_limit()
            
            respuesta = client.models.generate_content(
                model=modelo,
                contents=prompt
            )
            return respuesta.text
        except APIError as e:
            mensaje = str(e)
            if "RESOURCE_EXHAUSTED" in mensaje or "UNAVAILABLE" in mensaje or "503" in mensaje or "429" in mensaje:
                # Si es error de rate limit (429), esperar más tiempo
                tiempo_espera = espera * (2 if "429" in mensaje else 1)
                print(f"⚠️ API sobrecargada o límite de rate excedido (intento {i+1}/{intentos}). Reintentando en {tiempo_espera}s...")
                time.sleep(tiempo_espera)
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

# FUNCIÓN DE PROCESAMIENTO GEMINI: ESTRUCTURACIÓN
def estructurar_con_reintentos(texto):
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

    MANEJO DE IMÁGENES Y FIGURAS:
    - Si encuentras referencias a imágenes o figuras (sin las etiquetas <IMG_BLOCK>), evaluar si son esenciales para comprender el contenido
    - Si la imagen es esencial: agregar una descripción técnica breve del contenido visual
    - Si la imagen NO es esencial: omitirla completamente del fragmento
    - NO incluir frases como "ver figura 3.2" o "como muestra la imagen" sin contexto

    MANEJO DE BLOQUES DE CÓDIGO:
    - Si encuentras bloques de código (sin las etiquetas <CODIGO_BLOCK>), incluirlos siempre con su explicación
    - Corregir la sintaxis del código si está incorrecta
    - Agregar descripción técnica mínima que explique qué hace el código
    - Mantener la relación entre el código y el texto académico circundante

    NOTACIÓN MATEMÁTICA:
    - Usar exclusivamente texto plano o símbolos Unicode para expresiones matemáticas
    - Escribir conjuntos y expresiones de forma directa: usar formato plano como Σ = a, b, c
    - NO usar ningún tipo de notación especial, markup matemático o formatos externos
    - Las expresiones deben ser legibles directamente en texto sin renderizado adicional

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
    
    return manejar_reintentos(prompt, MODELO)


# PROCESAMIENTO PRINCIPAL CON FILTRO

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
if archivos_a_estructurar:
    tiempo_estimado = len(archivos_a_estructurar) * REQUEST_INTERVAL
    print(f"⏱️  Tiempo estimado mínimo: {tiempo_estimado/60:.1f} minutos ({REQUEST_INTERVAL:.1f}s por archivo)\n")

# --- 2. Procesamiento de los archivos filtrados ---
for idx, ruta_entrada in enumerate(archivos_a_estructurar, 1):
    nombre_archivo = os.path.basename(ruta_entrada)
    print(f"\n🧩 [{idx}/{len(archivos_a_estructurar)}] Procesando: {nombre_archivo}")

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