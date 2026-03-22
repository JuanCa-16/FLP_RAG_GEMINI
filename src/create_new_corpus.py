import os
import re
import json
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from itertools import count

ID_GENERATOR = count(1)

# CONFIGURACIÓN

RUTA_PADRE = os.path.abspath(os.path.dirname(__file__)) #src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # root

ARCHIVO_SALIDA = os.path.join(RUTA_PADRE, "embeddings", "new_corpus.jsonl")
CARPETA_ENTRADA_GEMINI = os.path.join(BASE_DIR, "NEW_TXT_GEMINI")
CARPETA_EJEMPLOS = os.path.join(BASE_DIR, "NEW_TXT_EJEMPLOS")
CARPETA_GIT_FLP = os.path.join(BASE_DIR, "NEW_GIT_FLP_GEMINI")
ARCHIVO_CONSOLIDADO = os.path.join(BASE_DIR, "txt_global.txt")

# STOPWORDS

try:
    STOP_WORDS_ES = set(stopwords.words('spanish'))
except LookupError:
    nltk.download('stopwords')
    STOP_WORDS_ES = set(stopwords.words('spanish'))

STOP_WORDS_ES.update({
    "bien", "entonces", "ahora", "ok", "vale", "correcto", "listo", "o sea", "etc", "por ejemplo",
    "sí", "si", "no", "sea", "ser", "estar", "haber", "hacer", "decir", "tener", "aquí", "allí", "esto", "eso",
    "muchachos", "ustedes", "gracias", "por su", "profesor", "curso", "bueno", "pues", "debe", "cierto", "ej", 
    "digamos", "partir", "utilizar", "va", "pronto", "acá", "mencioné", "ejemplo", "quiero", "recuerden", "siguientes",
    "sencillamente", "vamos", "perdón", "próximo", "sé", "vamos", "voy", "miren", "así"
})

STOP_WORDS_ES = list(STOP_WORDS_ES)


# ====================================
# FUNCIONES AUXILIARES
# ====================================

def limpiar_texto(texto: str) -> str:
    """Limpieza de texto sin eliminar signos útiles ni palabras acentuadas."""
    texto = re.sub(r'\s+', ' ', texto.strip())
    texto = re.sub(r'[""''´`]', "'", texto)
    texto = re.sub(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚüÜñÑ.,;:!?()\[\]\-–_/"% ]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()


def cargar_vectorizador_consolidado(path: str) -> TfidfVectorizer:
    """Carga el vectorizador TF-IDF entrenado con el corpus consolidado."""
    with open(path, "r", encoding="utf-8") as f:
        texto_consolidado = f.read()
    texto_consolidado = limpiar_texto(texto_consolidado)
    vect = TfidfVectorizer(stop_words=STOP_WORDS_ES)
    vect.fit([texto_consolidado])
    return vect


def extraer_palabras_clave(texto: str, vectorizador: TfidfVectorizer, top_n: int = 8) -> List[str]:
    """Extrae las palabras clave más relevantes usando TF-IDF."""
    if len(texto.split()) < 5:
        return []
    try:
        tfidf = vectorizador.transform([texto])
        scores = tfidf.toarray()[0]
        indices = scores.argsort()[-top_n:][::-1]
        return [vectorizador.get_feature_names_out()[i] for i in indices]
    except Exception:
        return []


def extraer_metadata(contenido: str) -> tuple[Dict[str, str], str]:
    """
    Extrae la metadata del contenido, soportando dos formatos:
    1. --- METADATA ---
    2. ---\nMETADATA\n---
    
    Retorna: (diccionario_metadata, contenido_sin_metadata)
    """
    metadata = {}
    
    # Intentar ambos formatos de metadata
    # Formato 1: --- METADATA ---
    meta_match = re.search(r'---\s*METADATA\s*---([\s\S]*)$', contenido)
    
    # Formato 2: ---\nMETADATA\n---
    if not meta_match:
        meta_match = re.search(r'---\s*\n\s*METADATA\s*\n\s*---([\s\S]*)$', contenido)
    
    if meta_match:
        meta_texto = meta_match.group(1)
        for linea in meta_texto.split("\n"):
            linea = linea.strip()
            if ":" in linea:
                k, v = linea.split(":", 1)
                metadata[k.strip()] = v.strip()
        # Retornar contenido sin la metadata
        contenido_sin_metadata = contenido[:meta_match.start()].strip()
    else:
        contenido_sin_metadata = contenido.strip()
    
    return metadata, contenido_sin_metadata


def dividir_por_fragmentos(contenido: str) -> List[str]:
    """
    Divide el contenido en fragmentos delimitados por ####.
    - Ignora la línea del delimitador ####
    - Retorna lista de fragmentos de texto
    """
    fragmentos = []
    acumulado = []
    lineas = contenido.splitlines()
    
    for linea in lineas:
        linea_limpia = linea.strip()
        
        # Detectar el delimitador de fragmento ####
        if linea_limpia.startswith("####"):
            # Si hay contenido acumulado, guardarlo
            if acumulado:
                texto_fragmento = "\n".join(acumulado).strip()
                if texto_fragmento:
                    fragmentos.append(texto_fragmento)
            
            # Reiniciar acumulador (no incluir la línea ####)
            acumulado = []
        else:
            # Agregar línea al fragmento actual
            if linea_limpia:  # Solo agregar líneas no vacías
                acumulado.append(linea)
    
    # Agregar el último fragmento si existe
    if acumulado:
        texto_fragmento = "\n".join(acumulado).strip()
        if texto_fragmento:
            fragmentos.append(texto_fragmento)
    
    return fragmentos


# ====================================
# PROCESAMIENTO PRINCIPAL
# ====================================

def procesar_carpeta(carpeta: str, vectorizador: TfidfVectorizer, tipo_fuente: str) -> List[Dict]:
    """
    Procesa todos los archivos .txt de una carpeta.
    
    Args:
        carpeta: Ruta de la carpeta a procesar
        vectorizador: Vectorizador TF-IDF para palabras clave
        tipo_fuente: Tipo de fuente para clasificación (ej: "concepto", "codigo", "github")
    
    Returns:
        Lista de fragmentos procesados con su metadata
    """
    fragmentos_global = []
    
    if not os.path.exists(carpeta):
        print(f"⚠️  Carpeta no encontrada: {carpeta}")
        return fragmentos_global
    
    archivos_procesados = 0
    
    for archivo in os.listdir(carpeta):
        if not archivo.endswith(".txt"):
            continue
        
        ruta = os.path.join(carpeta, archivo)
        
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                contenido = f.read()
        except Exception as e:
            print(f"   ❌ No se pudo leer {archivo}: {e}")
            continue
        
        # 1. Extraer metadata
        metadata, contenido_sin_metadata = extraer_metadata(contenido)
        
        # 1.1 Si la metadata NO tiene NOMBRE_DOCUMENTO, agregarlo
        # Esto es importante para archivos video* que no lo incluyen en su metadata original
        if "NOMBRE_DOCUMENTO" not in metadata:
            metadata["NOMBRE_DOCUMENTO"] = archivo
        
        # 2. Dividir en fragmentos por ####
        fragmentos = dividir_por_fragmentos(contenido_sin_metadata)
        
        if not fragmentos:
            print(f"   ⚠️  {archivo}: No se generaron fragmentos")
            continue
        
        print(f"   📄 {archivo}: {len(fragmentos)} fragmentos extraídos")
        
        # 3. Procesar cada fragmento
        for fragmento_texto in fragmentos:
            if not fragmento_texto.strip():
                continue
            
            # Limpiar texto para palabras clave
            texto_limpio = limpiar_texto(fragmento_texto)
            
            # Extraer palabras clave
            palabras_clave = extraer_palabras_clave(texto_limpio, vectorizador)
            
            # Agregar fragmento al resultado
            fragmentos_global.append({
                "id": int(next(ID_GENERATOR)),
                "tipo": tipo_fuente,  # Usar el tipo pasado como parámetro
                "contenido": fragmento_texto,
                "metadata": metadata,
                "palabras_clave": palabras_clave,
                "titulo": os.path.splitext(archivo)[0]
            })
        
        archivos_procesados += 1
    
    print(f"   ✅ {archivos_procesados} archivos procesados")
    return fragmentos_global


# ====================================
# EJECUCIÓN PRINCIPAL
# ====================================

print("=" * 60)
print("🧠 GENERACIÓN DE CORPUS UNIFICADO PARA RAG")
print("=" * 60)

print("\n📊 Cargando vectorizador global...")
vectorizador_global = cargar_vectorizador_consolidado(ARCHIVO_CONSOLIDADO)
print("   ✅ Vectorizador cargado")

resultado = []

# ====================================
# PROCESAMIENTO POR CARPETAS
# ====================================

print("\n" + "=" * 60)
print("⚙️  PROCESANDO CARPETAS")
print("=" * 60)

# 1. Carpeta TXT_GEMINI (tipo: "concepto" como en el original)
print("\n📁 Procesando: TXT_GEMINI")
fragmentos_gemini = procesar_carpeta(
    CARPETA_ENTRADA_GEMINI, 
    vectorizador_global, 
    tipo_fuente="concepto"  # ← Tipo original de procesar_archivos()
)
resultado.extend(fragmentos_gemini)
print(f"   📊 Total acumulado: {len(resultado)} fragmentos")

# 2. Carpeta TXT_EJEMPLOS (tipo: "CODIGO" como en el original)
print("\n📁 Procesando: TXT_EJEMPLOS")
fragmentos_ejemplos = procesar_carpeta(
    CARPETA_EJEMPLOS, 
    vectorizador_global, 
    tipo_fuente="codigo"  # ← Tipo original de procesar_ejemplos()
)
resultado.extend(fragmentos_ejemplos)
print(f"   📊 Total acumulado: {len(resultado)} fragmentos")

# 3. Carpeta GIT_FLP_GEMINI (tipo: "github" como en el original)
print("\n📁 Procesando: GIT_FLP_GEMINI")
fragmentos_github = procesar_carpeta(
    CARPETA_GIT_FLP, 
    vectorizador_global, 
    tipo_fuente="github"  # ← Tipo original de procesar_github()
)
resultado.extend(fragmentos_github)
print(f"   📊 Total acumulado: {len(resultado)} fragmentos")

# ====================================
# GUARDAR RESULTADO
# ====================================

print("\n" + "=" * 60)
print("💾 GUARDANDO CORPUS")
print("=" * 60)

# Crear directorio de salida si no existe
os.makedirs(os.path.dirname(ARCHIVO_SALIDA), exist_ok=True)

print(f"\n💾 Guardando {len(resultado)} fragmentos en formato JSONL...")
with open(ARCHIVO_SALIDA, "w", encoding="utf-8") as f:
    for frag in resultado:
        f.write(json.dumps(frag, ensure_ascii=False) + "\n")

print(f"   ✅ Archivo guardado: {ARCHIVO_SALIDA}")

# ====================================
# RESUMEN FINAL
# ====================================

print("\n" + "=" * 60)
print("📊 RESUMEN FINAL")
print("=" * 60)

# Contar por tipo
conteo_por_tipo = {}
for frag in resultado:
    tipo = frag.get("tipo", "desconocido")
    conteo_por_tipo[tipo] = conteo_por_tipo.get(tipo, 0) + 1

print(f"\n📈 Fragmentos por tipo:")
for tipo, cantidad in sorted(conteo_por_tipo.items()):
    print(f"   • {tipo}: {cantidad} fragmentos")

print(f"\n✨ Total de fragmentos generados: {len(resultado)}")
print(f"📁 Archivo de salida: {os.path.basename(ARCHIVO_SALIDA)}")
print("\n" + "=" * 60)
print("✅ PROCESO COMPLETADO EXITOSAMENTE")
print("=" * 60)