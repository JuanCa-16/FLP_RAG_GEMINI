# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import numpy as np
from numpy.linalg import norm
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
import time
from google.genai.errors import ServerError

# ==========================================================
# 1. CONFIGURACIÓN y CARGA GLOBAL
# ==========================================================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Inicialización global del cliente y carga de datos
# Esto se hace una sola vez al iniciar el servidor (optimización clave)
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    # Si la clave API falla, se puede manejar el error aquí o dejar que el servidor falle al inicio
    print(f"Error al inicializar el cliente Gemini: {e}")
    client = None

MODEL_ID = "gemini-embedding-001"
GENERATIVE_MODEL = "gemini-2.5-flash"
ruta_embeddings = "corpus_con_embeddings.jsonl"

print("✅ Cargando embeddings existentes...")
try:
    # Se carga el DataFrame una sola vez
    docs_df = pd.read_json(ruta_embeddings, lines=True)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo {ruta_embeddings}. Asegúrese de que esté en el directorio correcto.")
    docs_df = None
except Exception as e:
    print(f"Error al cargar el DataFrame: {e}")
    docs_df = None

# Inicialización de FastAPI
app = FastAPI(
    title="API de RAG para Fundamentos de Compilación",
    description="Servicio de preguntas y respuestas basado en Retrieval Augmented Generation (RAG) con Gemini."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Luego puedes restringir a tu frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Modelo Pydantic para la entrada de la API
class Consulta(BaseModel):
    pregunta: str
    top_n: int = 1 # Parámetro opcional para elegir cuántos documentos usar

# ==========================================================
# 2. FUNCIONES DE RAG (Adaptadas de tu script)
# ==========================================================

def encontrar_documento_relevante(consulta: str, dataframe: pd.DataFrame, modelo: str, top_n: int = 3):
    """Encuentra los top N documentos más relevantes para la consulta dada."""
    if client is None:
        raise HTTPException(status_code=500, detail="El cliente Gemini no se inicializó correctamente.")
    
    # ... [TU LÓGICA ORIGINAL PARA EMBEDDING Y CÁLCULO DE SIMILITUD] ...
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
    # Los embeddings en el dataframe deben ser convertidos a np.array si están guardados como listas
    doc_embeddings_np = np.array(dataframe['embeddings'].tolist())
    
    # Multiplicación matricial para cálculo eficiente de similitud
    similitudes_np = np.dot(doc_embeddings_np, normed_embedding)

    # argsort() da los índices en orden ascendente, [::-1] invierte (descendente), [:top_n] toma los N primeros
    indices_top_n = np.argsort(similitudes_np)[::-1][:top_n]

    # Preparamos la lista de resultados
    resultados = []
    for indice in indices_top_n:
        resultados.append({
            "documento": dataframe.iloc[indice]["contenido"],
            "titulo": dataframe.iloc[indice]["titulo"],
            "similitud": similitudes_np[indice]
        })

    # Retornamos la lista de resultados
    return resultados


def generar_respuesta(consulta: str, contexto: str):
    """Genera una respuesta usando el modelo generativo y el contexto recuperado."""
    if client is None:
        raise HTTPException(status_code=500, detail="El cliente Gemini no se inicializó correctamente.")
        
    prompt = f"""
    Tu rol: Eres un asistente AI, con personalidad de profesor universitario experto en **Fundamentos de interpretacion y compilacion de lenguajes de programacion**. Tu misión es educar y profundizar temas técnicos.

    Tu tarea: Analizar exhaustivamente el **CONTEXTO** proporcionado (fragmentos de documentos) para responder a la **PREGUNTA** del usuario.

    Pautas INELUDIBLES para tu respuesta:

    ### I. Estructura y Estilo (Profesor Universitario)
    1.  **Tono y Claridad:** Mantén un tono **profesoral, formal y amigable**. Usa un lenguaje preciso, pero descompón las ideas complejas  en términos sencillos y accesibles para un estudiante que busca claridad.
    2.  **Concisión Técnica:** La explicación debe ser **completa y detallada**, pero **nunca redundante**. Evita la extensión innecesaria; ve directo al concepto técnico.
    3.  **Introducción:** Comienza con una breve introducción que valide la importancia del tema para la Ingeniería o Ciencias de la Computación.

    ### II. Uso del Contexto (Prioridad Máxima)
    1.  **Fidelidad al Contexto:** Utiliza **exclusivamente** la información contenida en el CONTEXTO. No inventes, ni busques, ni uses conocimiento externo.
    2.  **Gestión de Información Faltante:** Si la PREGUNTA no puede ser respondida completamente o en absoluto con el CONTEXTO, debes indicarlo claramente con una frase profesional como: "La información específica sobre [TEMA FALTANTE] no se encuentra en los documentos proporcionados."

    PREGUNTA: {consulta}

    CONTEXTO: {contexto}
    """

    # ... [TU LÓGICA ORIGINAL PARA GENERACIÓN CON REINTENTOS] ...
    respuesta_final = None
    max_retries = 5
    retry_count = 0

    while respuesta_final is None and retry_count < max_retries:
        try:
            respuesta = client.models.generate_content(
                model=GENERATIVE_MODEL,
                contents=prompt
            )
            respuesta_final = respuesta.text
            
        except ServerError as e:
            if '503 UNAVAILABLE' in str(e):
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 2**retry_count 
                    time.sleep(wait_time) 
                else:
                    raise HTTPException(status_code=503, detail="Se agotó el número máximo de reintentos. No se pudo obtener una respuesta del modelo (Error 503).") from e
            else:
                raise HTTPException(status_code=500, detail=f"Error inesperado del servidor: {e}") from e
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error inesperado al generar contenido: {e}") from e

    return respuesta_final


# ==========================================================
# 3. ENDPOINT DE LA API
# ==========================================================

@app.post("/rag/responder")
async def responder_pregunta(data: Consulta):
    """
    Endpoint principal para obtener una respuesta RAG.
    """
    if docs_df is None or docs_df.empty:
        raise HTTPException(status_code=500, detail="El corpus de embeddings no está cargado o está vacío.")

    consulta = data.pregunta
    top_n = data.top_n
    
    # 1. Recuperación de documentos
    try:
        documentos_relevantes = encontrar_documento_relevante(consulta, docs_df, MODEL_ID, top_n=top_n)
    except Exception as e:
        # La excepción ya fue manejada y convertida a HTTPException en la función,
        # pero es bueno tener un manejo genérico.
        raise HTTPException(status_code=500, detail=f"Error en la fase de Recuperación de Documentos (Embeddings): {e}")

    # 2. Concatenar el contexto
    contexto_combinado = "\n\n--- FUENTE ADICIONAL ---\n\n".join([
        f"Fuente: {doc['titulo']}\nContenido: {doc['documento']}" 
        for doc in documentos_relevantes
    ])

    # 3. Generación de la respuesta
    try:
        respuesta_final = generar_respuesta(consulta, contexto_combinado)
    except Exception as e:
        # La excepción ya fue manejada y convertida a HTTPException en la función.
        # Es para asegurar que cualquier fallo se propague como un error de API.
        raise e
        
    # 4. Retornar el resultado estructurado
    return {
        "pregunta": consulta,
        "respuesta": respuesta_final,
        "documentos_usados": [
            {"titulo": doc['titulo'], "similitud": f"{doc['similitud']:.4f}"} 
            for doc in documentos_relevantes
        ]
    }