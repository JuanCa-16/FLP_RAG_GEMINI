import os
import re
import json
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

# ==========================================================
# CONFIGURACIÓN (sin cambios)
# ==========================================================
RUTA_PADRE = os.path.abspath(os.path.dirname(__file__))
CARPETA_ENTRADA = os.path.join(RUTA_PADRE, "TXT_GEMINI")
ARCHIVO_SALIDA = os.path.join(RUTA_PADRE, "corpus_con_ejemplos.jsonl")
CARPETA_EJEMPLOS = os.path.join(RUTA_PADRE, "EJEMPLOS_TXT")
ARCHIVO_CONSOLIDADO = os.path.join(RUTA_PADRE, "txt_global.txt")
MIN_PALABRAS = 60
MAX_PALABRAS = 100

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

# ==========================================================
# FUNCIONES BASE (sin cambios)
# ==========================================================
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

# ==========================================================
# PROCESAMIENTO DE BLOQUES (Lógica Original: Archivos NO 'video')
# ==========================================================
def dividir_bloques(texto: str) -> List[Dict]:
    """Divide el texto usando los tags <CODIGO_BLOCK_START>, <IMG_BLOCK_START>, etc."""
    fragmentos = []
    # Nota: Se ha simplificado ligeramente para manejar bloques sin tags en el texto,
    # aunque la lógica es la misma que la original.
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
# PROCESAMIENTO DE BLOQUES (Lógica Original MODIFICADA: Archivos NO 'video')
# ==========================================================
def dividir_bloques_por_hash(texto: str) -> List[Dict]:
    """
    Divide el texto en bloques basados en las líneas que comienzan con '#####',
    tratando todos los fragmentos resultantes como de 'tipo': 'concepto'.
    Se detiene antes de la sección '--- METADATA ---'.
    """
    fragmentos = []

    # 1. Eliminar la parte del texto antes del primer '---' si existe (cabecera inicial)
    match_inicio = re.search(r'---\s*?\n', texto)
    if match_inicio:
        texto = texto[match_inicio.end():].strip()
    # Si no hay '---', el texto entero es el primer bloque a dividir

    lineas = texto.splitlines()
    acumulado = []
    
    for linea in lineas:
        linea_limpia = linea.strip()
        
        # Parar si llegamos a la METADATA (puede estar al final del archivo)
        if linea_limpia.startswith("--- METADATA"):
            break
            
        # Ignorar las líneas de solo '---'
        if linea_limpia == "---":
            continue

        # Detectar el delimitador de bloque '#####'
        if linea_limpia.startswith("#####"):
            # Si encontramos un nuevo encabezado, guardamos el bloque anterior
            if acumulado:
                texto_bloque = " ".join(acumulado).strip()
                if texto_bloque:
                    # Usamos limpiar_texto al final
                    fragmentos.append({"tipo": "concepto", "texto": limpiar_texto(texto_bloque)})
            
            # Limpiamos el acumulador y NO incluimos la línea '#####' en el bloque.
            acumulado = []
            
        elif linea_limpia: # Si la línea no es un '#####' y no está vacía
            acumulado.append(linea)

    # Añadir el último bloque si existe
    if acumulado:
        texto_bloque = " ".join(acumulado).strip()
        if texto_bloque:
            fragmentos.append({"tipo": "concepto", "texto": limpiar_texto(texto_bloque)})
    
    # Si no se generó ningún fragmento, procesar todo el texto restante como un concepto único
    if not fragmentos and texto.strip():
        return [{"tipo": "concepto", "texto": limpiar_texto(texto.strip())}]
        
    return fragmentos

# ==========================================================
# PROCESAMIENTO DE BLOQUES (Lógica Nueva: Archivos 'video')
# ==========================================================

def dividir_por_hashtag(texto: str) -> List[Dict]:
    """
    Divide el texto en bloques basados en las líneas que comienzan con '#',
    ignorando la línea con el '#' en sí.
    """
    fragmentos = []
    
    # 1. Eliminar la parte del texto antes del primer '---'
    match_inicio = re.search(r'---\s*?\n', texto)
    if match_inicio:
        texto = texto[match_inicio.end():].strip()
    else:
        return [{"tipo": "concepto", "texto": limpiar_texto(texto)}]


    lineas = texto.splitlines()
    acumulado = []
    
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
                    fragmentos.append({"tipo": "concepto", "texto": limpiar_texto(texto_bloque)})
            
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
            fragmentos.append({"tipo": "concepto", "texto": limpiar_texto(texto_bloque)})
    
    return fragmentos


# ==========================================================
# AGRUPACIÓN DE PÁRRAFOS Y UNIÓN (sin cambios)
# ==========================================================
def agrupar_parrafos(fragmentos: List[Dict]) -> List[Dict]:
    # ... (cuerpo de la función agrupar_parrafos es el mismo que en tu código original)
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
    # ... (cuerpo de la función unir_fragmentos_pequenos es el mismo que en tu código original)
    nuevos = []
    buffer = ""

    for frag in fragmentos:
        if frag.get("tipo") in ("codigo", "imagen"):
            if buffer:
                # Si hay buffer pendiente y el anterior es concepto, unir
                if nuevos and nuevos[-1]["tipo"] == "concepto":
                    nuevos[-1]["texto"] += " " + buffer.strip()
                buffer = ""
            nuevos.append(frag)
            continue
            
        texto = frag["texto"].strip()
        if len(texto.split()) < min_palabras:
            buffer += " " + texto
        else:
            if buffer:
                frag["texto"] = buffer.strip() + " " + frag["texto"]
                buffer = ""
            nuevos.append(frag)

    if buffer and nuevos:
        if nuevos[-1]["tipo"] == "concepto":
            nuevos[-1]["texto"] += " " + buffer.strip()
        elif not nuevos:
            return [{"tipo": "concepto", "texto": buffer.strip()}]
            
    if buffer and not nuevos:
        return [{"tipo": "concepto", "texto": buffer.strip()}]
        
    return nuevos

# ==========================================================
# PIPELINE PRINCIPAL (MODIFICADO CON LÓGICA CONDICIONAL)
# ==========================================================
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
            # Lógica nueva: división por líneas que empiezan con #
            fragmentos = dividir_por_hashtag(contenido)
            logica_usada = "Hashtag (#)"
        else:
            # Lógica original: división por tags de bloque (<CODIGO_BLOCK_START>)
            # fragmentos = dividir_bloques(contenido)
            # logica_usada = "Tags (<...BLOCK_START>)"
            fragmentos = dividir_bloques_por_hash(contenido) # ⬅️ CAMBIO AQUÍ
            logica_usada = "Delimitador (#####)"
        
        # 3. Agrupación y unión de párrafos (igual para ambos)
        #fragmentos = agrupar_parrafos(fragmentos)
        #fragmentos = unir_fragmentos_pequenos(fragmentos, MIN_PALABRAS) 

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


        # ===== EXPLICACION =====
        expl_match = re.search(
            r'######## EXPLICACION ########([\s\S]*?)########',
            contenido
        )

        if not expl_match:
            continue  # ejemplo mal formado

        explicacion = limpiar_texto(expl_match.group(1))
        if not explicacion:
            continue

        fragmentos_global.append({
            "tipo": "CODIGO",
            "contenido": explicacion,
            "metadata": {
                **metadata,
                "codigo": codigo
            },
            "palabras_clave": extraer_palabras_clave(explicacion, vectorizador),
            "titulo": os.path.splitext(archivo)[0]
        })

    return fragmentos_global

# ==========================================================
# EJECUCIÓN (sin cambios)
# ==========================================================
print("🧠 Cargando vectorizador global...")
vectorizador_global = cargar_vectorizador_consolidado(ARCHIVO_CONSOLIDADO)

print("⚙️ Procesando archivos...")

resultado = []

# Procesar conceptos (TXT_GEMINI)
resultado.extend(
    procesar_archivos(CARPETA_ENTRADA, vectorizador_global)
)

# Procesar ejemplos (TXT_EJEMPLOS)
resultado.extend(
    procesar_ejemplos(CARPETA_EJEMPLOS, vectorizador_global)
)
print(f"💾 Guardando {len(resultado)} fragmentos en formato JSONL...")
with open(ARCHIVO_SALIDA, "w", encoding="utf-8") as f:
    for frag in resultado:
        f.write(json.dumps(frag, ensure_ascii=False) + "\n")

print(f"✅ Proceso completado. Se generaron {len(resultado)} fragmentos en {ARCHIVO_SALIDA}")