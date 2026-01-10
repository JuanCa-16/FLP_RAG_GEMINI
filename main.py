import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
import pandas as pd
import numpy as np

load_dotenv()

MODEL_ID = "gemini-embedding-001"
BATCH_SIZE = 100 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
#RUTA_CORPUS = "corpus.jsonl" 
#RUTA_CORPUS_EMBEDDINGS = "corpus_con_embeddings.jsonl"
RUTA_CORPUS = "src/embeddings/corpus_con_ejemplos.jsonl" 
RUTA_CORPUS_EMBEDDINGS = "src/embeddings/corpus_con_ejemplos_embeddings.jsonl"



client = genai.Client(api_key=GEMINI_API_KEY)


# ----------------------------------------------------------------------
# 1. FUNCIÓN DE PROCESAMIENTO POR LOTES
# ----------------------------------------------------------------------

def generate_batch_embeddings(client, model_id, contents, task_type, output_dim, batch_size=BATCH_SIZE):
    """
    Genera embeddings para cada objeto en el Corpus, dividiéndolos en lotes.

    Args:
        client: El cliente de la API de GenAI.
        model_id (str): El ID del modelo a usar.
        contents (list[str]): Lista de textos (contenido de los documentos).
        task_type (str): Tipo de tarea de embedding (ej. "RETRIEVAL_DOCUMENT").
        output_dim (int): Dimensionalidad de salida.
        batch_size (int): Número de documentos por lote.

    Returns:
        list: Lista plana de todos los embeddings generados.
    """
    all_embeddings = []
    total_contents = len(contents)
    
    # Itera sobre la lista de contenidos, tomando fragmentos del tamaño del lote
    for i in range(0, total_contents, batch_size):
        batch = contents[i:i + batch_size]
        print(f"    -> Procesando lote {i//batch_size + 1}/{total_contents//batch_size + 1} ({len(batch)} documentos)")
        
        # Llama a la API de forma nativa con el lote
        try:
            response = client.models.embed_content(
                model=model_id,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=output_dim
                )
            )
            
            # Procesa cada embedding en el lote (normalización)
            for embedding_value in response.embeddings:
                # Convierte a numpy array para la normalización
                v = np.array(embedding_value.values)
                # Normaliza y convierte de nuevo a lista
                normalized_v = (v / np.linalg.norm(v)).tolist()
                all_embeddings.append(normalized_v)
                
        except Exception as e:
            # Manejo básico de errores y reintento con espera (backoff)
            print(f"❌ Error al procesar el lote {i}: {e}. Reintentando en 5 segundos...")
            time.sleep(5)
            # Para simplificar, podrías reintentar solo el lote fallido o detener el proceso
            # Aquí, simplemente se detiene para evitar bucles infinitos en un ejemplo
            raise e 
            
    return all_embeddings

# ----------------------------------------------------------------------
# 2. PROCESO PRINCIPAL
# ----------------------------------------------------------------------

# Cargar la base de conocimiento desde corpus.jsonl
documentos = []
try:
    with open(RUTA_CORPUS, "r", encoding="utf-8") as f:
        for linea in f:
            documentos.append(json.loads(linea.strip()))
except FileNotFoundError:
    print(f"🔴 Error: El archivo '{RUTA_CORPUS}' no fue encontrado.")
    exit()

docs_df = pd.DataFrame(documentos)

# ----------------------------------------------------------------------
# 3. Generar o cargar embeddings
# ----------------------------------------------------------------------

if os.path.exists(RUTA_CORPUS_EMBEDDINGS):
    print("✅ Cargando embeddings existentes desde archivo...")
    docs_df = pd.read_json(RUTA_CORPUS_EMBEDDINGS, lines=True)
else:
    print("⚙️ Generando embeddings por lotes...")

    # Extrae el contenido de todos los documentos en una sola lista
    all_contents = docs_df["contenido"].tolist()
    
    # Llama a la función de procesamiento por lotes
    try:
        all_embeddings = generate_batch_embeddings(
            client=client, 
            model_id=MODEL_ID, 
            contents=all_contents,
            task_type="RETRIEVAL_DOCUMENT",
            output_dim=768 # Mantenemos la dimensionalidad de 10 que usaste
        )

        # Asigna la lista completa de embeddings al DataFrame
        docs_df["embeddings"] = all_embeddings
        
        # Guardamos embeddings para no tener que recalcularlos después
        docs_df.to_json(RUTA_CORPUS_EMBEDDINGS, orient="records", lines=True, force_ascii=False)
        print(f"\n💾 Embeddings guardados en {RUTA_CORPUS_EMBEDDINGS}")

    except Exception as e:
        print(f"\n🛑 Falló la generación de embeddings: {e}")
        # Muestra una pequeña parte del DataFrame para inspección
        print(f"Estado del DataFrame antes de fallar:\n{docs_df.head(2)}")
        exit()


# 4. Mostrar el DataFrame (ejemplo)
print("\n" + "="*50)
print(f"Total de documentos procesados: {len(docs_df)}")
print(f"Ejemplo de dimensionalidad de embedding: {len(docs_df.iloc[0]['embeddings'])}")
print("="*50)
print("\n**Documentos (Primeras 5 filas):**\n")
print(docs_df[['titulo', 'contenido']].head())