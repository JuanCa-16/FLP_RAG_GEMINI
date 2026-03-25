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


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CARPETA_ENTRADA = os.path.join(BASE_DIR, "data", "txt", "raw", "NEW")
CARPETA_SALIDA = os.path.join(BASE_DIR,  "data", "txt", "processed", "GEMINI_3_FLASH","GEMINI_EJEMPLO_COD")
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
        print(f"   ⏱️  Esperando {tiempo_espera:.1f}s para respetar límite de rate...")
        time.sleep(tiempo_espera)
    
    last_request_time = time.time()


MODELO = "gemini-3-flash-preview"

print(f"🌟 Script de documentación de código fuente iniciado.")
print(f"   Modelo: Gemini 1.5 Flash")
print(f"   Rate Limit: {MAX_REQUESTS_PER_MINUTE} peticiones/minuto (~{REQUEST_INTERVAL:.1f}s entre peticiones)")
print(f"   Carpeta de Entrada: {CARPETA_ENTRADA}")
print(f"   Carpeta de Salida: {CARPETA_SALIDA}")
print("-" * 50)

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
                print(f"   ⚠️  Reintento {i+1}/{intentos} tras error API. Esperando {tiempo_espera}s...")
                time.sleep(tiempo_espera)
            elif "NOT_FOUND" in msg or "400" in msg:
                print(f"   ❌ Error fatal de configuración de modelo: {e}")
                return ""
            else:
                print(f"   ❌ Error API no controlado: {e}")
                return ""
        except Exception as e:
            print(f"   ❌ Error no controlado de la librería: {e}")
            return ""
    
    print("   ❌ Falló después de varios intentos.")
    return ""

# ----------------------------
# PROMPT MEJORADO
# ----------------------------
def generar_documentacion(codigo):
    """
    Genera documentación estructurada para código fuente.
    La IA NO genera metadata, solo procesa el código.
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

    ESTRUCTURA DE FRAGMENTOS DE CÓDIGO:
    Cada fragmento que contenga código debe incluir:
    1. Propósito (1-2 líneas): Qué problema resuelve o qué concepto ilustra
    2. Código en bloque markdown con especificador del lenguaje correcto
    3. Conceptos clave: Lista breve de términos técnicos (3-5 términos separados por comas)
    4. Aplicación práctica: Descripción concreta de cuándo/cómo se usa

    VARIEDAD LÉXICA:
    - Usar verbos variados: implementa, demuestra, ilustra, aplica, resuelve, ejemplifica
    - NO repetir "Ejemplo de..." al inicio de cada fragmento
    - Incluir nombres de funciones clave y estructuras sintácticas relevantes

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

    Ejemplo 1 - Código Racket documentado:

    ENTRADA:
    "Función para verificar si lista vacía:
    (define vacia? (lambda (lst) (null? lst)))"

    SALIDA:
    ####
    Verificación de listas vacías mediante el predicado `null?`:

    ```racket
    (define vacia?
    (lambda (lst)
        (null? lst)))
    ```

    **Conceptos clave**
    `predicados`, `listas`, `valores booleanos`, `funciones de orden superior`

    **Aplicación**
    Esta función se utiliza como condición base en algoritmos recursivos que procesan listas, permitiendo detectar cuándo se ha alcanzado el final de la estructura.
    Retorna `#t` si la lista está vacía, `#f` en caso contrario.

    ---

    Ejemplo 2 - Código Python:

    ENTRADA:
    "def suma(a,b):
    return a+b
    calcula suma de dos numeros"

    SALIDA:
    ####
    Función básica para sumar dos números:

    ```python
    def suma(a, b):
        return a + b
    ```

    **Conceptos clave**
    `funciones`, `parámetros`, `return`, `operadores aritméticos`

    **Aplicación**
    Operación fundamental utilizada en cálculos matemáticos y procesamiento de datos numéricos.

    
    CONTENIDO A PROCESAR:
    {codigo}
"""
    return manejar_reintentos(prompt, MODELO)

# ----------------------------
# CONSTRUIR METADATA
# ----------------------------
def construir_metadata(carpeta_tema, nombre_archivo, lenguaje="racket"):
    """
    Construye el bloque de metadata que se agregará al final del documento.
    Esta función maneja la metadata fuera del prompt de la IA.
    """
    return f"""
--- METADATA ---
NOMBRE_DOCUMENTO: {carpeta_tema}/{nombre_archivo}
LENGUAJE: {lenguaje}
FUENTE: CODIGO"""

# ----------------------------
# DETECTAR LENGUAJE
# ----------------------------
def detectar_lenguaje(nombre_archivo):
    """
    Detecta el lenguaje de programación basándose en la extensión del archivo.
    """
    extension = os.path.splitext(nombre_archivo)[1].lower()
    
    lenguajes = {
        '.rkt': 'racket',
        '.scm': 'scheme',
        '.py': 'python',
        '.js': 'javascript',
        '.java': 'java',
        '.c': 'c',
        '.cpp': 'c++',
        '.rs': 'rust',
    }
    
    return lenguajes.get(extension, 'desconocido')

# ----------------------------
# CONTAR ARCHIVOS TOTALES
# ----------------------------
def contar_archivos_totales():
    """Cuenta el total de archivos de código a procesar"""
    total = 0
    for carpeta_tema in os.listdir(CARPETA_ENTRADA):
        ruta_tema = os.path.join(CARPETA_ENTRADA, carpeta_tema)
        if os.path.isdir(ruta_tema):
            archivos = (
                glob.glob(os.path.join(ruta_tema, "*.rkt")) +
                glob.glob(os.path.join(ruta_tema, "*.scm"))
            )
            total += len(archivos)
    return total

# ----------------------------
# PROCESAMIENTO
# ----------------------------
print("\nIniciando procesamiento de archivos de código...\n")

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

for carpeta_tema in os.listdir(CARPETA_ENTRADA):
    ruta_tema = os.path.join(CARPETA_ENTRADA, carpeta_tema)

    if not os.path.isdir(ruta_tema):
        continue

    archivos = (
        glob.glob(os.path.join(ruta_tema, "*.rkt")) +
        glob.glob(os.path.join(ruta_tema, "*.scm"))
    )
    
    if not archivos:
        continue
    
    print(f"\n📁 Procesando carpeta: {carpeta_tema}")

    for ruta_archivo in archivos:
        contador_global += 1
        nombre_archivo = os.path.basename(ruta_archivo)
        print(f"\n   📄 [{contador_global}/{total_archivos}] Procesando: {nombre_archivo}")

        codigo = leer_archivo_normalizado(ruta_archivo)
        if not codigo or not codigo.strip():
            print(f"      ⚠️ Archivo vacío o no legible")
            archivos_con_error += 1
            continue

        # Detectar lenguaje
        lenguaje = detectar_lenguaje(nombre_archivo)

        # PASO 1: Gemini genera la documentación (sin metadata)
        contenido_procesado = generar_documentacion(codigo)
        
        if not contenido_procesado or not contenido_procesado.strip():
            print(f"      ⚠️ Gemini no pudo procesar el código")
            contenido_procesado = f"""####
ERROR: No se pudo procesar el archivo con Gemini

**Código original:**
```{lenguaje}
{codigo[:500]}...
```
"""
            archivos_con_error += 1

        # PASO 2: Agregar metadata al final
        metadata = construir_metadata(carpeta_tema, nombre_archivo, lenguaje)
        documento_final = f"{contenido_procesado.strip()}\n\n{metadata}"

        # PASO 3: Guardar archivo
        nombre_salida = f"{carpeta_tema}_{os.path.splitext(nombre_archivo)[0]}.txt"
        ruta_salida = os.path.join(CARPETA_SALIDA, nombre_salida)
        
        with open(ruta_salida, "w", encoding="utf-8") as f:
            f.write(documento_final)

        print(f"      ✅ Guardado: {nombre_salida}")
        archivos_procesados += 1

print("\n" + "=" * 50)
print("✅ Proceso completado exitosamente")
print(f"📂 Archivos guardados en: {CARPETA_SALIDA}")
print(f"\n📊 Estadísticas:")
print(f"   ✅ Procesados correctamente: {archivos_procesados}")
print(f"   ❌ Con errores: {archivos_con_error}")
print(f"   📝 Total: {archivos_procesados + archivos_con_error}")