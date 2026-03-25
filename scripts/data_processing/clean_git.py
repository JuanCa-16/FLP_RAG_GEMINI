import os
import glob
import json
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




BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CARPETA_ENTRADA = os.path.join(BASE_DIR, "data", "txt", "raw", "NEW")
CARPETA_SALIDA = os.path.join(BASE_DIR,  "data", "txt", "processed", "GEMINI_3_FLASH","GEMINI_GIT")
ARCHIVO_METADATA_JSON = os.path.join(BASE_DIR, "data", "metadata","tematicas.json")  # Ruta al JSON

os.makedirs(CARPETA_SALIDA, exist_ok=True)

# ----------------------------
# CONTROL DE RATE LIMITING
# ----------------------------
# Gemini 1.5 Flash: 15 peticiones por minuto
MAX_REQUESTS_PER_MINUTE = 5
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
        print(f"      ⏱️  Esperando {tiempo_espera:.1f}s para respetar límite de rate...")
        time.sleep(tiempo_espera)
    
    last_request_time = time.time()


MODELO = "gemini-3-flash-preview"

print(f"🌟 Script de procesamiento de archivos web iniciado.")
print(f"   Modelo: Gemini 1.5 Flash")
print(f"   Rate Limit: {MAX_REQUESTS_PER_MINUTE} peticiones/minuto (~{REQUEST_INTERVAL:.1f}s entre peticiones)")
print(f"   Carpeta de Entrada: {CARPETA_ENTRADA}")
print(f"   Carpeta de Salida: {CARPETA_SALIDA}")
print("-" * 50)

# ----------------------------
# CARGAR METADATA DESDE JSON
# ----------------------------
def cargar_metadata_json(ruta_json):
    """
    Carga el archivo JSON con la metadata de las clases.
    Retorna el diccionario o None si hay error.
    """
    try:
        with open(ruta_json, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  ADVERTENCIA: No se encontró el archivo {ruta_json}")
        return None
    except json.JSONDecodeError:
        print(f"⚠️  ADVERTENCIA: Error al leer el JSON {ruta_json}")
        return None
    except Exception as e:
        print(f"⚠️  ADVERTENCIA: Error inesperado al cargar JSON: {e}")
        return None

# Cargar metadata al inicio
METADATA_CLASES = cargar_metadata_json(ARCHIVO_METADATA_JSON)

def obtener_indice_json(carpeta_clase):
    """
    Extrae el número de clase de la carpeta y lo mapea al índice del JSON.
    
    Mapeo:
    - Clase 1 → índice 0
    - Clase 2-8 → índice (número - 1)
    - Clase 9 → índice 7
    - Clase 10, 11 → índice 8
    - Clase 12 → índice 9
    - Clase 13 → índice 10
    """
    import re
    match = re.search(r'[Cc]lase\s*(\d+)', carpeta_clase)
    
    if not match:
        return None
    
    numero_clase = int(match.group(1))
    
    # Mapeo especial
    if numero_clase <= 8:
        return str(numero_clase - 1)
    elif numero_clase == 9:
        return "7"
    elif numero_clase in [10, 11]:
        return "8"
    elif numero_clase == 12:
        return "9"
    elif numero_clase == 13:
        return "10"
    
    return None


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
    # Construir NOMBRE_DOCUMENTO (sin extensión)
    nombre_sin_ext = os.path.splitext(nombre_archivo)[0]
    nombre_documento = f"{carpeta_clase}_{nombre_sin_ext}"
    
    # Verificar que METADATA_CLASES esté cargado
    if METADATA_CLASES is None:
        return f"""---
METADATA
---
CORTE: No especificado (JSON no cargado)
TEMATICA: {carpeta_clase}
COMPETENCIA: No especificada
RESULTADO_APRENDIZAJE: No especificado
NIVEL_DIFICULTAD: No especificada
FUENTE: GIT
NOMBRE_DOCUMENTO: {nombre_documento}
URL: {url}"""
    
    # Obtener índice del JSON
    indice = obtener_indice_json(carpeta_clase)
    
    # Obtener metadata del JSON si existe
    if indice and indice in METADATA_CLASES:
        meta = METADATA_CLASES[indice]
        return f"""---
METADATA
---
CORTE: {meta['CORTE']}
TEMATICA: {meta['TEMATICA']}
COMPETENCIA: {meta['COMPETENCIA']}
RESULTADO_APRENDIZAJE: {meta['RESULTADO_APRENDIZAJE']}
NIVEL_DIFICULTAD: {meta['NIVEL_DIFICULTAD']}
FUENTE: GIT
NOMBRE_DOCUMENTO: {nombre_documento}
URL: {url}"""
    else:
        # Fallback si no se encuentra en el JSON
        return f"""---
METADATA
---
CORTE: No especificado
TEMATICA: {carpeta_clase}
COMPETENCIA: No especificada
RESULTADO_APRENDIZAJE: No especificado
NIVEL_DIFICULTAD: No especificada
FUENTE: GIT
NOMBRE_DOCUMENTO: {nombre_documento}
URL: {url}"""
# ----------------------------
# REINTENTOS CON RATE LIMITING
# ----------------------------
def manejar_reintentos(prompt, modelo, intentos=3, espera=10):
    """Función genérica para manejar la llamada a la API con reintentos y rate limiting."""
    for i in range(intentos):
        try:
            # Esperar antes de hacer la petición para respetar rate limit
            esperar_rate_limit()
            
            resp = client.models.generate_content(
                model=modelo,
                contents=prompt
            )
            
            # Extraer solo las partes de texto para evitar warnings
            texto_completo = ""
            for parte in resp.candidates[0].content.parts:
                if hasattr(parte, 'text'):
                    texto_completo += parte.text
            
            return texto_completo if texto_completo else resp.text
            
        except APIError as e:
            msg = str(e)
            if any(x in msg for x in ["RESOURCE_EXHAUSTED", "UNAVAILABLE", "503", "429"]):
                # Si es error de rate limit (429), esperar más tiempo
                tiempo_espera = espera * (2 if "429" in msg else 1)
                print(f"      ⚠️  Reintento {i+1}/{intentos} tras error API. Esperando {tiempo_espera}s...")
                time.sleep(tiempo_espera)
            elif "NOT_FOUND" in msg or "400" in msg:
                print(f"      ❌ Error fatal de configuración de modelo: {e}")
                return ""
            else:
                print(f"      ❌ Error API no controlado: {e}")
                return ""
        except Exception as e:
            print(f"      ❌ Error no controlado de la librería: {e}")
            return ""
    
    print("      ❌ Falló después de varios intentos.")
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
    [CONTEXTO]
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

    RECONSTRUCCIÓN DE CÓDIGO Y DIAGRAMAS:
    - Si el código aparece con formato UN CARÁCTER POR LÍNEA, reconstruir completamente
    - Si encuentras diagramas Mermaid malformados, corregir la sintaxis
    - Si encuentras diagramas GraphViz malformados, corregir la sintaxis
    - Verificar que la sintaxis sea válida después de reconstruir
    - Mantener la indentación estándar del lenguaje correspondiente

    IDENTIFICACIÓN DE LENGUAJES:
    - Racket: (define ...), lambda, sintaxis con paréntesis S-expressions
    - Python: def, class, import, indentación significativa
    - JavaScript: function, const, let, var, =>
    - Mermaid: graph TD, flowchart, sequenceDiagram, etc.
    - GraphViz: digraph, graph, node, edge
    - Otros: identificar por sintaxis característica

    CORRECCIÓN DE DIAGRAMAS MERMAID:
    - Formato correcto: graph TD o graph LR
    - Nodos con corchetes: A["Texto del nodo"]
    - Flechas: A --> B o A --> |etiqueta| B
    - Eliminar caracteres extraños como (0 0 0 0 0 0) que no son válidos

    CORRECCIÓN DE DIAGRAMAS GRAPHVIZ:
    - Formato correcto: digraph nombre { ... }
    - Nodos: A [label="texto"]
    - Aristas: A -> B [label="etiqueta"]

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

    [EJEMPLOS]

    Ejemplo 1 - Diagrama Mermaid malformado:

    ENTRADA:
    "graph TD     A["C3 (0 0 0 0 0 0)"] B[C2] A-->B"

    SALIDA:
    ####
    **Diagrama de dependencias entre componentes**
    ```mermaid
    graph TD
        A["C3"]
        B["C2"]
        A --> B
    ```

    Este diagrama muestra la relación de dependencia donde el componente `C3` apunta hacia `C2`.

    ---

    Ejemplo 2 - Código web con múltiples lenguajes:

    ENTRADA:
    "Aqui ejemplo python:
    d
    e
    f
    suma(a,b): return a+b

    Y diagram:
    graph TD A-->B"

    SALIDA:
    ####
    **Función básica de suma en Python**
    ```python
    def suma(a, b):
        return a + b
    ```

    Esta función toma dos parámetros y retorna su suma aritmética.

    ####
    **Diagrama de flujo simple**
    ```mermaid
    graph TD
        A --> B
    ```

    Representa una relación directa entre los nodos `A` y `B`.


    CONTENIDO A PROCESAR:
    {contenido_original}

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
    contenido_limpio = re.sub(r'--- METADATA ---[\s\S]*$', '', contenido_organizado).strip()
    
    # Construir metadata usando la función dedicada
    metadata = construir_metadata(carpeta_clase, nombre_archivo, url)
    
    # Ensamblar documento final
    documento_final = f"{contenido_limpio}\n\n{metadata}"
    
    return documento_final

# ----------------------------
# CONTAR ARCHIVOS TOTALES
# ----------------------------
def contar_archivos_totales():
    """Cuenta el total de archivos .txt a procesar"""
    total = 0
    for carpeta_clase in os.listdir(CARPETA_ENTRADA):
        ruta_clase = os.path.join(CARPETA_ENTRADA, carpeta_clase)
        if os.path.isdir(ruta_clase):
            archivos_txt = glob.glob(os.path.join(ruta_clase, "*.txt"))
            total += len(archivos_txt)
    return total

# ----------------------------
# PROCESAMIENTO
# ----------------------------
print("\nIniciando procesamiento de archivos web...\n")

# Calcular tiempo estimado
total_archivos = contar_archivos_totales()
if total_archivos > 0:
    tiempo_estimado = total_archivos * REQUEST_INTERVAL
    print(f"⏱️  Total de archivos a procesar: {total_archivos}")
    print(f"⏱️  Tiempo estimado mínimo: {tiempo_estimado/60:.1f} minutos ({REQUEST_INTERVAL:.1f}s por archivo)\n")
    print("=" * 50)

archivos_procesados = 0
archivos_con_error = 0
contador_global = 0

for carpeta_clase in os.listdir(CARPETA_ENTRADA):
    ruta_clase = os.path.join(CARPETA_ENTRADA, carpeta_clase)

    if not os.path.isdir(ruta_clase):
        continue

    print(f"\n📁 Procesando clase: {carpeta_clase}")

    archivos_txt = glob.glob(os.path.join(ruta_clase, "*.txt"))

    if not archivos_txt:
        print(f"   ⚠️  No se encontraron archivos .txt\n")
        continue

    for ruta_archivo in archivos_txt:
        contador_global += 1
        nombre_archivo = os.path.basename(ruta_archivo)
        print(f"\n   📄 [{contador_global}/{total_archivos}] Procesando: {nombre_archivo}")

        contenido = leer_archivo_normalizado(ruta_archivo)
        if not contenido or not contenido.strip():
            print(f"      ❌ Archivo vacío o no legible")
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
            print(f"      ⚠️  Gemini no pudo procesar el contenido")
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
            print(f"      ✅ Guardado con metadata: {nombre_salida}")
            archivos_procesados += 1
        else:
            print(f"      ⚠️  ADVERTENCIA: Metadata no encontrada en {nombre_salida}")
            archivos_con_error += 1

print("\n" + "=" * 50)
print("✅ Proceso completado exitosamente")
print(f"📂 Archivos guardados en: {CARPETA_SALIDA}")
print(f"\n📊 Estadísticas:")
print(f"   ✅ Procesados correctamente: {archivos_procesados}")
print(f"   ❌ Con errores: {archivos_con_error}")
print(f"   📝 Total: {archivos_procesados + archivos_con_error}")