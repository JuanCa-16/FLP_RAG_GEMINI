import os
import re
import json
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk


#ESTE NO INCLUYE LOS EJEMPLOS (LA IDEA ES BORRAR ESTE ARCHIVO A FUTURO)

# ==========================================================
# CONFIGURACIÓN (sin cambios)
# ==========================================================
RUTA_PADRE = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # root

CARPETA_ENTRADA = os.path.join(BASE_DIR, "TXT_GEMINI")
ARCHIVO_SALIDA = os.path.join(RUTA_PADRE, "embeddings", "corpus.jsonl")
ARCHIVO_CONSOLIDADO = os.path.join(BASE_DIR, "txt_global.txt")


# ==========================================================
# STOPWORDS (Español) (sin cambios)
# ==========================================================
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

# ==========================================================
# EJECUCIÓN (sin cambios)
# ==========================================================
print("🧠 Cargando vectorizador global...")
vectorizador_global = cargar_vectorizador_consolidado(ARCHIVO_CONSOLIDADO)

print("⚙️ Procesando archivos...")
resultado = procesar_archivos(CARPETA_ENTRADA, vectorizador_global)

print(f"💾 Guardando {len(resultado)} fragmentos en formato JSONL...")
with open(ARCHIVO_SALIDA, "w", encoding="utf-8") as f:
    for frag in resultado:
        f.write(json.dumps(frag, ensure_ascii=False) + "\n")

print(f"✅ Proceso completado. Se generaron {len(resultado)} fragmentos en {ARCHIVO_SALIDA}")