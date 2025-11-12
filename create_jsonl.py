import os
import re
import json
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

# ==========================================================
# CONFIGURACIÓN
# ==========================================================
RUTA_PADRE = os.path.abspath(os.path.dirname(__file__))
CARPETA_ENTRADA = os.path.join(RUTA_PADRE, "TXT_METADATA")
ARCHIVO_SALIDA = os.path.join(RUTA_PADRE, "corpus.jsonl")
ARCHIVO_CONSOLIDADO = os.path.join(RUTA_PADRE, "txt_global.txt")
MIN_PALABRAS = 60
MAX_PALABRAS = 100

# ==========================================================
# STOPWORDS (Español)
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

# ==========================================================
# FUNCIONES BASE
# ==========================================================
def limpiar_texto(texto: str) -> str:
    """Limpieza de texto sin eliminar signos útiles ni palabras acentuadas."""
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

# ==========================================================
# PROCESAMIENTO DE BLOQUES
# ==========================================================
def dividir_bloques(texto: str) -> List[Dict]:
    fragmentos = []
    bloques = re.split(r'(<CODIGO_BLOCK_START>|<IMG_BLOCK_START>|<CODIGO_BLOCK_END>|<IMG_BLOCK_END>)', texto)

    tipo_actual = "concepto"
    acumulado = []

    for bloque in bloques:
        bloque = bloque.strip()
        if not bloque:
            continue

        if bloque in ("<CODIGO_BLOCK_START>", "<IMG_BLOCK_START>"):
            if acumulado:
                fragmentos.append({"tipo": tipo_actual, "texto": limpiar_texto(" ".join(acumulado))})
                acumulado = []
            tipo_actual = "codigo" if "CODIGO" in bloque else "imagen"

        elif bloque in ("<CODIGO_BLOCK_END>", "<IMG_BLOCK_END>"):
            if acumulado:
                fragmentos.append({"tipo": tipo_actual, "texto": limpiar_texto(" ".join(acumulado))})
                acumulado = []
            tipo_actual = "concepto"

        else:
            acumulado.append(bloque)

    if acumulado:
        fragmentos.append({"tipo": tipo_actual, "texto": limpiar_texto(" ".join(acumulado))})

    return fragmentos

# ==========================================================
# AGRUPACIÓN DE PÁRRAFOS
# ==========================================================
def agrupar_parrafos(fragmentos: List[Dict]) -> List[Dict]:
    resultado = []

    for frag in fragmentos:
        palabras = frag["texto"].split()
        if frag["tipo"] in ("codigo", "imagen"):
            resultado.append(frag)
            continue

        inicio = 0
        temp_result = []

        while inicio < len(palabras):
            fin = min(inicio + MAX_PALABRAS, len(palabras))
            texto_parcial = " ".join(palabras[inicio:fin])

            ultimo_punto = texto_parcial.rfind('.')
            if ultimo_punto != -1:
                palabras_hasta_punto = texto_parcial[:ultimo_punto + 1].split()
                if len(palabras_hasta_punto) >= MIN_PALABRAS:
                    fin = inicio + len(palabras_hasta_punto)

            texto_fragmento = " ".join(palabras[inicio:fin])
            temp_result.append({"tipo": "concepto", "texto": texto_fragmento})
            inicio = fin

        if len(temp_result) > 1 and len(temp_result[-1]["texto"].split()) < MIN_PALABRAS:
            temp_result[-2]["texto"] += " " + temp_result[-1]["texto"]
            temp_result.pop(-1)

        resultado.extend(temp_result)

    return resultado

def unir_fragmentos_pequenos(fragmentos: List[Dict], min_palabras: int = 50) -> List[Dict]:
    nuevos = []
    buffer = ""

    for frag in fragmentos:
        texto = frag["texto"].strip()
        if len(texto.split()) < min_palabras:
            buffer += " " + texto
        else:
            if buffer:
                frag["texto"] = buffer.strip() + " " + frag["texto"]
                buffer = ""
            nuevos.append(frag)

    if buffer and nuevos:
        nuevos[-1]["texto"] += " " + buffer.strip()

    return nuevos

# ==========================================================
# PIPELINE PRINCIPAL
# ==========================================================
def procesar_archivos(carpeta: str, vectorizador: TfidfVectorizer) -> List[Dict]:
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

        meta_match = re.search(r'--- METADATA ---([\s\S]*)$', contenido)
        metadata = {}
        if meta_match:
            meta_texto = meta_match.group(1)
            for linea in meta_texto.split("\n"):
                if ":" in linea:
                    k, v = linea.split(":", 1)
                    metadata[k.strip()] = v.strip()
            contenido = contenido[:meta_match.start()]

        fragmentos = dividir_bloques(contenido)
        fragmentos = agrupar_parrafos(fragmentos)
        fragmentos = unir_fragmentos_pequenos(fragmentos, MIN_PALABRAS)

        print(f"📄 {archivo}: {len(fragmentos)} fragmentos generados")

        for frag in fragmentos:
            frag["contenido"] = frag.pop("texto")
            frag["metadata"] = metadata
            frag["palabras_clave"] = extraer_palabras_clave(frag["contenido"], vectorizador)
            frag["titulo"] = os.path.splitext(archivo)[0]  # nombre del archivo sin extensión
            fragmentos_global.append(frag)

    return fragmentos_global

# ==========================================================
# EJECUCIÓN
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
