import os
import json
import numpy as np
from numpy.linalg import norm
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
import time # Para pausar la ejecución
from google.genai.errors import ServerError

# ==========================================================
# CONFIGURACIÓN
# ==========================================================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

MODEL_ID = "gemini-embedding-001"
GENERATIVE_MODEL = "gemini-2.5-flash"
ruta_embeddings = "corpus_con_embeddings.jsonl"

# ==========================================================
# CARGAR EMBEDDINGS Y CONSULTA
# ==========================================================
print("✅ Cargando embeddings existentes...")
docs_df = pd.read_json(ruta_embeddings, lines=True)

#consulta = "Que es una operacion declarativa?"
#consulta = "Que es un ambiente?"
#consulta = "Diferencias entre ligadura y asignacion"
consulta = "Que es un dato? "


print(f"📘 Consulta del usuario: {consulta}\n")

# 3. Función para buscar el documento más relevante para una consulta
def encontrar_documento_relevante(consulta, dataframe, modelo):
    """Encuentra el documento más relevante para la consulta dada."""

    # Generamos el embedding para la consulta con RETRIEVAL_QUERY
    embedding_consulta = client.models.embed_content(
        model=modelo,
        contents=consulta,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY",output_dimensionality=768)
    )

    [embedding_reducido] = embedding_consulta.embeddings

    embedding_values_np = np.array(embedding_reducido.values)
    normed_embedding = embedding_values_np / np.linalg.norm(embedding_values_np)
    

    # Calculamos la similitud del coseno entre la consulta y todos los documentos
    similitudes = []
    for doc_embedding in dataframe.embeddings:
        # Similitud del coseno entre dos vectores
        similitud = np.dot(doc_embedding, normed_embedding)
        similitudes.append(similitud)

    # Encontramos el índice del documento con mayor similitud
    indice_mejor = np.argmax(similitudes)

    # Retornamos información sobre el mejor resultado
    return {
        "documento": dataframe.iloc[indice_mejor]["contenido"],
        "titulo": dataframe.iloc[indice_mejor]["titulo"],
        "similitud": similitudes[indice_mejor],
        "todos_scores": dict(zip(dataframe.titulo, similitudes))
    }

def generar_respuesta(consulta, contexto):
    """Genera una respuesta usando el modelo generativo y el contexto recuperado."""

    prompt = f"""
    Tu rol: Eres un asistente AI especializado en explicar y recomendar temas de programación y fundamentos de lenguajes a estudiantes universitarios.

    Tu tarea: Utiliza el CONTEXTO proporcionado para responder a la PREGUNTA del usuario.

    Pautas para tu respuesta:
    - Sé claro y simple: Explica cualquier idea complicada en términos fáciles de entender, sin perder lo tecnico y no tan extenso.
    - Sé amigable: Escribe en un tono explicativo como profesor universitario.
    - Sé completo: Construye una respuesta detallada utilizando toda la información relevante del contexto.
    - Céntrate en el tema: Si el contexto no contiene la respuesta, indica que la información no está disponible.

    PREGUNTA: {consulta}

    CONTEXTO: {contexto}
    """

    # Generamos la respuesta
    respuesta = client.models.generate_content(
        model=GENERATIVE_MODEL,
        contents=prompt
    )

    return respuesta.text


resultado = encontrar_documento_relevante(consulta, docs_df, MODEL_ID)

# Mostramos los resultados
print(f"Consulta: {consulta}\n")
print(f"Documento más relevante: {resultado['titulo']}")
print(f"concepto: {resultado['documento']}")
print(f"Puntuación de similitud: {resultado['similitud']:.4f}\n")

respuesta_final = None
max_retries = 5
retry_count = 0

while respuesta_final is None and retry_count < max_retries:
    try:
        print(f"🚀 Intentando generar respuesta... (Intento {retry_count + 1}/{max_retries})")
        # Aquí se llama a la función que puede fallar con 503
        respuesta_final = generar_respuesta(consulta, resultado["documento"])
        print("\n🧠 Respuesta generada:")
        print(respuesta_final)

    except ServerError as e:
        # Capturamos específicamente el error del servidor (como el 503)
        if '503 UNAVAILABLE' in str(e):
            retry_count += 1
            if retry_count < max_retries:
                # El backoff exponencial (2, 4, 8, 16, 32 segundos de espera)
                wait_time = 2**retry_count 
                print(f"\n⚠️ ERROR 503 UNAVAILABLE. Reintentando en {wait_time} segundos... (Intento {retry_count + 1}/{max_retries})")
                time.sleep(wait_time) 
            else:
                 print("\n❌ Se agotó el número máximo de reintentos. No se pudo obtener una respuesta (Error 503).")

        else:
            # Si es otro ServerError, mostramos el error y terminamos el bucle
            print(f"\n❌ Se produjo otro error de servidor: {e}")
            break

    except Exception as e:
        # Capturamos cualquier otro error inesperado
        print(f"\n❌ Se produjo un error inesperado: {e}")
        break