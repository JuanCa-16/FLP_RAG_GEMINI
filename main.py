import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
import pandas as pd
import numpy as np

# corpues con los embeddings 
load_dotenv()

MODEL_ID = "gemini-embedding-001"  # Gemini Embedding 2
BATCH_SIZE = 100 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
RUTA_PADRE = os.path.abspath(os.path.dirname(__file__))

RUTA_CORPUS = os.path.join(RUTA_PADRE, "src", "embeddings", "new_corpus.jsonl")
RUTA_CORPUS_EMBEDDINGS = os.path.join(RUTA_PADRE, "src", "embeddings", "new_corpus_embeddings.jsonl")

# ----------------------------
# CONTROL DE RATE LIMITING
# ----------------------------
# Gemini Embedding 2: 100 peticiones por minuto
MAX_REQUESTS_PER_MINUTE = 100
REQUEST_INTERVAL = 60.0 / MAX_REQUESTS_PER_MINUTE  # ~0.6 segundos entre peticiones
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
        time.sleep(tiempo_espera)
    
    last_request_time = time.time()


client = genai.Client(api_key=GEMINI_API_KEY)

print("=" * 70)
print("🧠 GENERACIÓN DE EMBEDDINGS CON GEMINI EMBEDDING 2")
print("=" * 70)
print(f"   Modelo: {MODEL_ID}")
print(f"   Rate Limit: {MAX_REQUESTS_PER_MINUTE} peticiones/minuto (~{REQUEST_INTERVAL:.2f}s entre peticiones)")
print(f"   Tamaño de lote: {BATCH_SIZE} documentos")
print(f"   Dimensionalidad: 768")
print("=" * 70)


# ----------------------------------------------------------------------
# 1. FUNCIÓN DE PROCESAMIENTO POR LOTES CON REINTENTOS
# ----------------------------------------------------------------------

def generate_batch_embeddings(client, model_id, contents, task_type, output_dim, batch_size=BATCH_SIZE, intentos=3):
    """
    Genera embeddings para cada objeto en el Corpus, dividiéndolos en lotes.
    Incluye lógica de reintentos y control de rate limiting.

    Args:
        client: El cliente de la API de GenAI.
        model_id (str): El ID del modelo a usar.
        contents (list[str]): Lista de textos (contenido de los documentos).
        task_type (str): Tipo de tarea de embedding (ej. "RETRIEVAL_DOCUMENT").
        output_dim (int): Dimensionalidad de salida (768 para Gemini Embedding 2).
        batch_size (int): Número de documentos por lote.
        intentos (int): Número máximo de reintentos por lote.

    Returns:
        list: Lista plana de todos los embeddings generados.
    """
    all_embeddings = []
    total_contents = len(contents)
    total_lotes = (total_contents + batch_size - 1) // batch_size
    
    print(f"\n📊 Total de documentos: {total_contents}")
    print(f"📊 Total de lotes: {total_lotes}")
    
    # Calcular tiempo estimado
    tiempo_estimado = total_lotes * REQUEST_INTERVAL
    print(f"⏱️  Tiempo estimado mínimo: {tiempo_estimado/60:.1f} minutos\n")
    
    # Itera sobre la lista de contenidos, tomando fragmentos del tamaño del lote
    for i in range(0, total_contents, batch_size):
        batch = contents[i:i + batch_size]
        numero_lote = i//batch_size + 1
        
        print(f"📦 Procesando lote {numero_lote}/{total_lotes} ({len(batch)} documentos)")
        
        # Intentar procesar el lote con reintentos
        lote_procesado = False
        for intento in range(intentos):
            try:
                # Esperar antes de hacer la petición para respetar rate limit
                esperar_rate_limit()
                
                # Llama a la API de forma nativa con el lote
                response = client.models.embed_content(
                    model=model_id,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=output_dim
                    )
                )
                
                # Procesa cada embedding en el lote (normalización)
                batch_embeddings = []
                for embedding_value in response.embeddings:
                    # Convierte a numpy array para la normalización
                    v = np.array(embedding_value.values)
                    # Normaliza y convierte de nuevo a lista
                    normalized_v = (v / np.linalg.norm(v)).tolist()
                    batch_embeddings.append(normalized_v)
                
                # Agregar embeddings del lote exitoso
                all_embeddings.extend(batch_embeddings)
                print(f"   ✅ Lote {numero_lote} procesado exitosamente")
                lote_procesado = True
                break  # Salir del bucle de reintentos
                
            except APIError as e:
                msg = str(e)
                
                # Manejo de errores según el tipo
                if any(x in msg for x in ["RESOURCE_EXHAUSTED", "UNAVAILABLE", "503", "429"]):
                    # Error de sobrecarga o rate limit
                    tiempo_espera = 10 * (2 ** intento)  # Backoff exponencial
                    if "429" in msg:
                        tiempo_espera = 20 * (2 ** intento)  # Espera más larga para rate limit
                    
                    print(f"      ⚠️  API sobrecargada o límite de rate (intento {intento+1}/{intentos})")
                    
                    if intento < intentos - 1:
                        print(f"      ⏳ Esperando {tiempo_espera}s antes de reintentar...")
                        time.sleep(tiempo_espera)
                    else:
                        print(f"      ❌ Lote {numero_lote} falló después de {intentos} intentos")
                        raise e
                        
                elif "NOT_FOUND" in msg or "400" in msg:
                    print(f"      ❌ Error fatal de configuración: {e}")
                    raise e
                else:
                    print(f"      ❌ Error API no controlado: {e}")
                    if intento < intentos - 1:
                        print(f"      ⏳ Reintentando en 5s...")
                        time.sleep(5)
                    else:
                        raise e
                        
            except Exception as e:
                print(f"      ❌ Error inesperado: {e}")
                if intento < intentos - 1:
                    print(f"      ⏳ Reintentando en 5s...")
                    time.sleep(5)
                else:
                    raise e
        
        if not lote_procesado:
            print(f"      🛑 Lote {numero_lote} no pudo procesarse. Deteniendo ejecución.")
            raise Exception(f"Fallo al procesar lote {numero_lote}")
            
    return all_embeddings


# ----------------------------------------------------------------------
# 2. PROCESO PRINCIPAL
# ----------------------------------------------------------------------

print("\n📂 Cargando corpus...")

# Cargar la base de conocimiento desde corpus.jsonl
documentos = []
try:
    with open(RUTA_CORPUS, "r", encoding="utf-8") as f:
        for linea in f:
            documentos.append(json.loads(linea.strip()))
    print(f"   ✅ {len(documentos)} documentos cargados desde {os.path.basename(RUTA_CORPUS)}")
except FileNotFoundError:
    print(f"   🔴 Error: El archivo '{RUTA_CORPUS}' no fue encontrado.")
    exit()

docs_df = pd.DataFrame(documentos)

# ----------------------------------------------------------------------
# 3. Generar o cargar embeddings
# ----------------------------------------------------------------------

if os.path.exists(RUTA_CORPUS_EMBEDDINGS):
    print("\n" + "=" * 70)
    print("✅ Embeddings ya existen. Cargando desde archivo...")
    print("=" * 70)
    docs_df = pd.read_json(RUTA_CORPUS_EMBEDDINGS, lines=True)
    print(f"   ✅ {len(docs_df)} documentos con embeddings cargados")
else:
    print("\n" + "=" * 70)
    print("⚙️  GENERANDO EMBEDDINGS")
    print("=" * 70)

    # Extrae el contenido de todos los documentos en una sola lista
    all_contents = docs_df["contenido"].tolist()
    
    # Llama a la función de procesamiento por lotes
    try:
        all_embeddings = generate_batch_embeddings(
            client=client, 
            model_id=MODEL_ID, 
            contents=all_contents,
            task_type="RETRIEVAL_DOCUMENT",
            output_dim=768, 
            intentos=3
        )

        # Asigna la lista completa de embeddings al DataFrame
        docs_df["embeddings"] = all_embeddings
        
        print("\n" + "=" * 70)
        print("💾 GUARDANDO EMBEDDINGS")
        print("=" * 70)
        
        # Guardamos embeddings para no tener que recalcularlos después
        docs_df.to_json(RUTA_CORPUS_EMBEDDINGS, orient="records", lines=True, force_ascii=False)
        print(f"   ✅ Embeddings guardados en: {os.path.basename(RUTA_CORPUS_EMBEDDINGS)}")

    except Exception as e:
        print("\n" + "=" * 70)
        print("🛑 FALLO EN LA GENERACIÓN DE EMBEDDINGS")
        print("=" * 70)
        print(f"   ❌ Error: {e}")
        print(f"\n📊 Estado del DataFrame antes de fallar:")
        print(docs_df[['id', 'titulo', 'tipo']].head(5))
        exit()


# ----------------------------------------------------------------------
# 4. RESUMEN Y VALIDACIÓN
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("📊 RESUMEN DEL CORPUS CON EMBEDDINGS")
print("=" * 70)

print(f"\n📈 Estadísticas generales:")
print(f"   • Total de documentos: {len(docs_df)}")
print(f"   • Dimensionalidad de embeddings: {len(docs_df.iloc[0]['embeddings'])}")

# Contar por tipo
if 'tipo' in docs_df.columns:
    print(f"\n📂 Distribución por tipo:")
    conteo_tipos = docs_df['tipo'].value_counts()
    for tipo, cantidad in conteo_tipos.items():
        print(f"   • {tipo}: {cantidad} documentos")

# Verificar que todos los embeddings están presentes
embeddings_validos = docs_df['embeddings'].apply(lambda x: len(x) == 768).sum()
print(f"\n✅ Embeddings válidos (768 dimensiones): {embeddings_validos}/{len(docs_df)}")

if embeddings_validos != len(docs_df):
    print(f"   ⚠️  ADVERTENCIA: {len(docs_df) - embeddings_validos} embeddings con dimensionalidad incorrecta")

print("\n" + "=" * 70)
print("📄 MUESTRA DE DOCUMENTOS (Primeras 5 filas)")
print("=" * 70)
print(docs_df[['id', 'tipo', 'titulo']].head())

print("\n" + "=" * 70)
print("✅ PROCESO COMPLETADO EXITOSAMENTE")
print("=" * 70)
print(f"\n📁 Archivo de salida: {os.path.basename(RUTA_CORPUS_EMBEDDINGS)}")
print(f"📍 Ruta completa: {RUTA_CORPUS_EMBEDDINGS}\n")