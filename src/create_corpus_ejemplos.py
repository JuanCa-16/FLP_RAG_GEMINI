import os
import re
import json
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk


# CONFIGURACIÓN

RUTA_PADRE = os.path.abspath(os.path.dirname(__file__)) #src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # root

ARCHIVO_SALIDA = os.path.join(RUTA_PADRE, "embeddings", "corpus_con_ejemplos.jsonl") # cambiar nombre si es con ejemplo o no y borrar abajo
CARPETA_ENTRADA = os.path.join(BASE_DIR, "TXT_METADATA")
CARPETA_ENTRADA_GEMINI = os.path.join(BASE_DIR, "TXT_GEMINI")
CARPETA_EJEMPLOS = os.path.join(BASE_DIR, "TXT_EJEMPLOS")
CARPETA_GIT_FLP= os.path.join(BASE_DIR, "GIT_FLP_GEMINI")
ARCHIVO_CONSOLIDADO = os.path.join(BASE_DIR, "txt_global.txt")
MIN_PALABRAS = 60
MAX_PALABRAS = 100


# STOPWORDS

try:
    STOP_WORDS_ES = set(stopwords.words('spanish'))
except LookupError:
    nltk.download('stopwords')
    STOP_WORDS_ES = set(stopwords.words('spanish'))

STOP_WORDS_ES.update({
    "bien", "entonces", "ahora", "ok", "vale", "correcto", "listo", "o sea", "etc", "por ejemplo",
    "sí", "si", "no", "sea", "ser", "estar", "haber", "hacer", "decir", "tener", "aquí", "allí", "esto", "eso",
    "muchachos", "ustedes", "gracias", "por su", "profesor", "curso", "bueno", "pues", "debe", "cierto", "ej", "digamos","partir","utilizar","va","pronto", "acá","mencioné", "ejemplo", "quiero", "recuerden","siguientes",
"sencillamente","vamos","perdón","próximo","sé","vamos","voy","miren","así"
})

STOP_WORDS_ES = list(STOP_WORDS_ES)


def limpiar_texto(texto: str) -> str:
    """Limpieza de texto sin eliminar signos útiles ni palabras acentuadas."""

    texto = re.sub(r'<CODIGO_BLOCK_START>|<CODIGO_BLOCK_END>|<IMG_BLOCK_START>|<IMG_BLOCK_END>', ' ', texto)

    texto = re.sub(r'\s+', ' ', texto.strip())
    texto = re.sub(r'[“”‘’´`]', "'", texto)
    texto = re.sub(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚüÜñÑ.,;:!?()\[\]\-–_/"% ]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()

def cargar_vectorizador_consolidado(path: str) -> TfidfVectorizer:
    with open(path, "r", encoding="utf-8") as f:
        texto_consolidado = f.read()
    texto_consolidado = limpiar_texto(texto_consolidado)
    vect = TfidfVectorizer(stop_words=STOP_WORDS_ES)
    vect.fit([texto_consolidado])
    return vect

def extraer_palabras_clave(texto: str, vectorizador: TfidfVectorizer, top_n: int = 8) -> List[str]:
    if len(texto.split()) < 5:
        return []
    try:
        tfidf = vectorizador.transform([texto])
        scores = tfidf.toarray()[0]
        indices = scores.argsort()[-top_n:][::-1]
        return [vectorizador.get_feature_names_out()[i] for i in indices]
    except Exception:
        return []

def dividir_bloques_por_hash(texto: str) -> List[Dict]:
    """
     Divide el texto en bloques delimitados por '#####'.
    - El primer bloque empieza desde la primera línea del archivo.
    - Cada '#####' inicia un nuevo bloque.
    - Se ignoran líneas de control (<IMG_BLOCK>, <CODIGO_BLOCK>, etc).
    - Se detiene antes de '--- METADATA ---'.
    """

    fragmentos = []
    acumulado = []
    lineas = texto.splitlines()

    BLOQUES_INVALIDOS = {
        "<IMG_BLOCK>", "</IMG_BLOCK>",
        "<CODIGO_BLOCK>", "</CODIGO_BLOCK>"
    }
    
    for linea in lineas:
        linea_limpia = linea.strip()
        
        # Parar si llegamos a la METADATA (puede estar al final del archivo)
        if linea_limpia.startswith("--- METADATA"):
            break
            
        # Ignorar las líneas de solo '---'
        if linea_limpia == "---":
            continue

         # Ignorar líneas de control
        if linea_limpia in BLOQUES_INVALIDOS:
            continue

        # Detectar el delimitador de bloque '#####'
        if linea_limpia.startswith("####"):
            # Si encontramos un nuevo encabezado, guardamos el bloque anterior
            if acumulado:
                texto_bloque = " ".join(acumulado).strip()
                if texto_bloque:
                    # Usamos limpiar_texto al final
                    fragmentos.append({"tipo": "concepto", "texto": texto_bloque})
            
            # Limpiamos el acumulador y NO incluimos la línea '#####' en el bloque.
            acumulado = []
            
        elif linea_limpia: # Si la línea no es un '#####' y no está vacía
            acumulado.append(linea)

    # Añadir el último bloque si existe
    if acumulado:
        texto_bloque = " ".join(acumulado).strip()
        if texto_bloque:
            fragmentos.append({"tipo": "concepto", "texto": texto_bloque})
    
    # Si no se generó ningún fragmento, procesar todo el texto restante como un concepto único
    if not fragmentos and texto.strip():
        return [{"tipo": "concepto", "texto": texto.strip()}]
        
    return fragmentos

def dividir_por_hashtag(texto: str) -> List[Dict]:
    """
    Divide el texto en bloques basados en las líneas que comienzan con '#',
    ignorando la línea con el '#' en sí.
    """
    fragmentos = []
    acumulado = []
    lineas = texto.splitlines()
    
    for linea in lineas:
        linea_limpia = linea.strip()
        
        # Ignorar las líneas de solo '---'
        if linea_limpia == "---":
            continue
            
        # Parar si llegamos a la METADATA
        if linea_limpia.startswith("--- METADATA"):
            break

        if linea_limpia.startswith("#"):
            # Si encontramos un nuevo encabezado, guardamos el bloque anterior
            if acumulado:
                texto_bloque = " ".join(acumulado).strip()
                if texto_bloque:
                    fragmentos.append({"tipo": "concepto", "texto": texto_bloque})
            
            # Limpiamos el acumulador y no incluimos la línea '#' en el bloque.
            acumulado = []

            # *** MODIFICACIÓN CLAVE: QUITAR los '#' e INCLUIR el resto de la línea
            titulo_limpio = re.sub(r'^#+\s*', '', linea_limpia)
            if titulo_limpio:
                 # Añadir el título como el primer elemento del nuevo bloque
                 acumulado.append(titulo_limpio)
            
        elif linea_limpia: # Si la línea no es un hashtag y no está vacía
            acumulado.append(linea)

    # Añadir el último bloque si existe
    if acumulado:
        texto_bloque = " ".join(acumulado).strip()
        if texto_bloque:
            fragmentos.append({"tipo": "concepto", "texto": texto_bloque})
    
    return fragmentos

def procesar_archivos(carpeta: str, vectorizador: TfidfVectorizer) -> List[Dict]:
    fragmentos_global = []

    for archivo in os.listdir(carpeta):
        if not archivo.endswith(".txt"):
            continue
            
        # 🌟 LÓGICA CONDICIONAL: Verificar el prefijo del archivo
        es_archivo_video = archivo.lower().startswith("video")

        ruta = os.path.join(carpeta, archivo)
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                contenido = f.read()
        except Exception as e:
            print(f"⚠️ No se pudo leer el archivo {archivo}: {e}")
            continue

        # 1. Extracción de METADATA
        meta_match = re.search(r'--- METADATA ---([\s\S]*)$', contenido)
        metadata = {}
        if meta_match:
            meta_texto = meta_match.group(1)
            for linea in meta_texto.split("\n"):
                if ":" in linea:
                    k, v = linea.split(":", 1)
                    metadata[k.strip()] = v.strip()
            contenido = contenido[:meta_match.start()]
            
        # 2. División en bloques basada en la condición
        if es_archivo_video:
            fragmentos = dividir_por_hashtag(contenido)
            logica_usada = "Hashtag (#)"
        else:
            fragmentos = dividir_bloques_por_hash(contenido) # ⬅️ CAMBIO AQUÍ
            logica_usada = "Delimitador (#####)"
    

        print(f"📄 {archivo}: {len(fragmentos)} fragmentos generados. Lógica: {logica_usada}")

        # 4. Enriquecimiento de fragmentos (igual para ambos)
        for frag in fragmentos:
            if not frag["texto"].strip():
                 continue
                 
            frag["contenido"] = frag.pop("texto")
            frag["metadata"] = metadata
            frag["palabras_clave"] = extraer_palabras_clave(frag["contenido"], vectorizador)
            frag["titulo"] = os.path.splitext(archivo)[0]
            fragmentos_global.append(frag)

    return fragmentos_global

def procesar_ejemplos(carpeta: str, vectorizador: TfidfVectorizer) -> List[Dict]:
    fragmentos_global = []

    for archivo in os.listdir(carpeta):
        if not archivo.endswith(".txt"):
            continue

        ruta = os.path.join(carpeta, archivo)
        with open(ruta, "r", encoding="utf-8") as f:
            contenido = f.read()

        # ===== METADATA =====
        meta_match = re.search(r'######## METADATA ########([\s\S]*?)########', contenido)
        metadata = {}
        if meta_match:
            for linea in meta_match.group(1).splitlines():
                if ":" in linea:
                    k, v = linea.split(":", 1)
                    metadata[k.strip()] = v.strip()

        # ===== CODIGO =====
        codigo_match = re.search(
            r'######## CODIGO ########([\s\S]*)$',
            contenido
        )
        codigo = ""
        if codigo_match:
            codigo = codigo_match.group(1).strip()

        # ===== RESUMEN =====
        resumen_match = re.search(
            r'######## RESUMEN ########([\s\S]*?)########',
            contenido
        )
        resumen = ""
        if resumen_match:
            resumen = limpiar_texto(resumen_match.group(1))

        # ===== CONCEPTOS =====
        conceptos_match = re.search(
            r'######## CONCEPTOS ########([\s\S]*?)########',
            contenido
        )
        conceptos = ""
        if conceptos_match:
            conceptos = limpiar_texto(conceptos_match.group(1))

        # ===== EXPLICACION =====
        explicacion_match = re.search(
            r'######## EXPLICACION ########([\s\S]*?)########',
            contenido
        )
        explicacion = ""
        if explicacion_match:
            explicacion = limpiar_texto(explicacion_match.group(1))

        # ===== CONCATENAR PARA EMBEDDING =====
        # Construye el texto completo que irá al embedding
        texto_completo = ""
        if resumen:
            texto_completo += resumen
        if conceptos:
            texto_completo += ". " + conceptos if texto_completo else conceptos
        if explicacion:
            texto_completo += ". " + explicacion if texto_completo else explicacion

        # Validar que hay contenido
        if not texto_completo.strip():
            print(f"⚠️ Advertencia: {archivo} no tiene contenido válido")
            continue

        fragmentos_global.append({
            "tipo": "CODIGO",
            "contenido": texto_completo,  # AHORA ES LA CONCATENACIÓN COMPLETA
            "metadata": {
                **metadata,
                "codigo": codigo,
                "resumen": resumen,  # Opcional: guardar por separado
                "conceptos": conceptos,  # Opcional: para filtrado posterior
            },
            "palabras_clave": extraer_palabras_clave(texto_completo, vectorizador),
            "titulo": os.path.splitext(archivo)[0]
        })

    return fragmentos_global

def procesar_github(carpeta: str, vectorizador: TfidfVectorizer) -> List[Dict]:
    """
    Procesa archivos de GitHub organizados por Gemini.
    Divide el contenido en chunks delimitados por '###'.
    Extrae metadata y la agrega al final de cada documento.
    """
    fragmentos_global = []

    for archivo in os.listdir(carpeta):
        if not archivo.endswith(".txt"):
            continue

        ruta = os.path.join(carpeta, archivo)
        
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                contenido = f.read()
        except Exception as e:
            print(f"⚠️ No se pudo leer el archivo {archivo}: {e}")
            continue
        # ===== EXTRACCIÓN DE METADATA DEL CONTENIDO =====
        meta_match = re.search(r'---\s*METADATA\s*---([\s\S]*?)$', contenido)
        metadata = {}
        
        if meta_match:
            meta_texto = meta_match.group(1)
            for linea in meta_texto.split("\n"):
                linea = linea.strip()
                if ":" in linea:
                    k, v = linea.split(":", 1)
                    metadata[k.strip()] = v.strip()
            # Eliminar la metadata del contenido a procesar
            contenido_sin_metadata = contenido[:meta_match.start()].strip()
        else:
            contenido_sin_metadata = contenido.strip()

        # ===== DIVISIÓN EN CHUNKS POR '###' =====
        # El contenido puede empezar directamente (sin ### inicial)
        # o puede empezar con ###
        
        chunks = []
        chunk_actual = []
        lineas = contenido_sin_metadata.split('\n')
        
        # Si la primera línea no es ###, estamos en el primer chunk
        en_primer_chunk = True
        
        for linea in lineas:
            linea_limpia = linea.strip()
            
            if linea_limpia == "###":
                # Guardar chunk anterior si existe
                if chunk_actual:
                    texto_chunk = "\n".join(chunk_actual).strip()
                    if texto_chunk:
                        chunks.append(texto_chunk)
                # Resetear para nuevo chunk
                chunk_actual = []
                en_primer_chunk = False
            else:
                # Agregar línea al chunk actual
                chunk_actual.append(linea)
        
        # Agregar último chunk
        if chunk_actual:
            texto_chunk = "\n".join(chunk_actual).strip()
            if texto_chunk:
                chunks.append(texto_chunk)
        
        print(f"📄 {archivo}: {len(chunks)} fragmentos generados (GitHub)")

        # ===== ENRIQUECIMIENTO DE FRAGMENTOS =====
        for chunk_texto in chunks:
            if not chunk_texto.strip():
                continue
            
            # Limpiar texto para palabras clave
            texto_limpio = limpiar_texto(chunk_texto)
            
            # Extraer palabras clave
            palabras_clave = extraer_palabras_clave(texto_limpio, vectorizador)
            
            fragmentos_global.append({
                "tipo": "github",
                "contenido": chunk_texto,
                "metadata": metadata,
                "palabras_clave": palabras_clave,
                "titulo": os.path.splitext(archivo)[0]
            })

    return fragmentos_global
# EJECUCIÓN

print("🧠 Cargando vectorizador global...")
vectorizador_global = cargar_vectorizador_consolidado(ARCHIVO_CONSOLIDADO)

print("⚙️ Procesando archivos...")

resultado = []

# Procesar conceptos (TXT_GEMINI)
resultado.extend(
    procesar_archivos(CARPETA_ENTRADA_GEMINI, vectorizador_global)
)

# Procesar ejemplos (TXT_EJEMPLOS)
resultado.extend(
    procesar_ejemplos(CARPETA_EJEMPLOS, vectorizador_global)
)

# Procesar ejemplos (TXT_EJEMPLOS)
resultado.extend(
    procesar_github(CARPETA_GIT_FLP, vectorizador_global)
)
print(f"💾 Guardando {len(resultado)} fragmentos en formato JSONL...")
with open(ARCHIVO_SALIDA, "w", encoding="utf-8") as f:
    for frag in resultado:
        f.write(json.dumps(frag, ensure_ascii=False) + "\n")

print(f"✅ Proceso completado. Se generaron {len(resultado)} fragmentos en {ARCHIVO_SALIDA}")