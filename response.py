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
consulta = "En que consiste un objeto"
#consulta = "Como se clasifican los lenguajes"


print(f"📘 Consulta del usuario: {consulta}\n")

# 3. Función para buscar el documento más relevante para una consulta
def encontrar_documento_relevante(consulta, dataframe, modelo, top_n=3):
    """Encuentra los top N documentos más relevantes para la consulta dada."""

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
        # Nota: Asume que los embeddings del dataframe ya están normalizados o los normaliza aquí si es necesario
        similitud = np.dot(doc_embedding, normed_embedding)
        similitudes.append(similitud)

    # Convertimos a array de numpy para encontrar los top N
    similitudes_np = np.array(similitudes)
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
    return {
        "top_documentos": resultados,
        "todos_scores": dict(zip(dataframe.titulo, similitudes))
    }


def generar_respuesta(consulta, contexto):
    """Genera una respuesta usando el modelo generativo y el contexto recuperado."""

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

    # Generamos la respuesta
    respuesta = client.models.generate_content(
        model=GENERATIVE_MODEL,
        contents=prompt
    )

    return respuesta.text


resultado = encontrar_documento_relevante(consulta, docs_df, MODEL_ID,top_n=1)

# 1. Concatenar el contexto de los documentos más relevantes
contexto_combinado = "\n\n--- FUENTE ADICIONAL ---\n\n".join([
    f"Fuente: {doc['titulo']}\nContenido: {doc['documento']}" 
    for doc in resultado['top_documentos']
])

# 2. Mostramos los resultados de la búsqueda
print(f"Consulta: {consulta}\n")
print("🔍 Documentos más relevantes (Top 3):")
for i, doc in enumerate(resultado['top_documentos']):
    # **CAMBIO CLAVE AQUÍ**
    # Creamos un snippet de 250 caracteres del contenido para la impresión en consola
    snippet_contenido = doc['documento'][:250].replace('\n', ' ')
    if len(doc['documento']) > 250:
        snippet_contenido += "..."
        
    print(f"  {i+1}. Título: {doc['titulo']} (Similitud: {doc['similitud']:.4f})")
    print(f"     Snippet: {snippet_contenido}")


print("\n--- Contexto Combinado Enviado al Modelo ---")
print(contexto_combinado[:500] + "...") # Imprimimos solo los primeros 500 caracteres del contexto
print("-------------------------------------------\n")


respuesta_final = None
max_retries = 5
retry_count = 0


while respuesta_final is None and retry_count < max_retries:
    try:
        print(f"🚀 Intentando generar respuesta con contexto combinado... (Intento {retry_count + 1}/{max_retries})")
        # Aquí se llama a la función que puede fallar con 503
        respuesta_final = generar_respuesta(consulta, contexto_combinado)
        print("\n🧠 Respuesta generada:")
        print(respuesta_final)
        
    except ServerError as e:
        if '503 UNAVAILABLE' in str(e):
            retry_count += 1
            if retry_count < max_retries:
                # Backoff exponencial
                wait_time = 2**retry_count 
                print(f"\n⚠️ ERROR 503 UNAVAILABLE. Reintentando en {wait_time} segundos... (Intento {retry_count + 1}/{max_retries})")
                time.sleep(wait_time) 
            else:
                print("\n❌ Se agotó el número máximo de reintentos. No se pudo obtener una respuesta (Error 503).")
        else:
            print(f"\n❌ Se produjo otro error de servidor: {e}")
            break
            
    except Exception as e:
        print(f"\n❌ Se produjo un error inesperado: {e}")
        break