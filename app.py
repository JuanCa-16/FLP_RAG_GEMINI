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
ruta_embeddings = "corpus_con_ejemplos_embeddings.jsonl"

docs_df_all = None
docs_df_pdf = None
docs_df_video = None
docs_df_codigos = None

print("✅ Cargando y filtrando embeddings existentes...")
try:
    # 1. Carga del DataFrame COMPLETO (se hace una sola vez)
    docs_df_todo = pd.read_json(ruta_embeddings, lines=True)
    
    # Asegurarse de que 'metadata' es un diccionario (o se puede acceder a 'FUENTE')
    def get_fuente(metadata):
        return metadata.get('FUENTE', '').upper() if isinstance(metadata, dict) else ''

    docs_df_todo['FUENTE'] = docs_df_todo['metadata'].apply(get_fuente)

    # 2. Filtrado eficiente para los DataFrames específicos
    docs_df_pdf = docs_df_todo[docs_df_todo['FUENTE'] == 'PDF'].copy()
    docs_df_video = docs_df_todo[docs_df_todo['FUENTE'] == 'VIDEO'].copy()
    docs_df_codigos = docs_df_todo[docs_df_todo['FUENTE'] == 'CODIGO'].copy()

    # Log de conteo para verificación
    print(f"  - Total de documentos cargados: {len(docs_df_todo)}")
    print(f"  - Documentos PDF disponibles: {len(docs_df_pdf)}")
    print(f"  - Documentos VIDEO disponibles: {len(docs_df_video)}")

except FileNotFoundError:
    print(f"Error: No se encontró el archivo {ruta_embeddings}.")
    docs_df_todo = docs_df_pdf = docs_df_video = None
except Exception as e:
    print(f"Error al cargar/filtrar el DataFrame: {e}")
    docs_df_todo = docs_df_pdf = docs_df_video = None


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
    
    if dataframe is None or dataframe.empty:
        # Esto será manejado por los endpoints, pero es una buena práctica aquí también
        raise HTTPException(status_code=404, detail="El corpus de documentos seleccionado está vacío.")
    
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
            "metadata": dataframe.iloc[indice]["metadata"],
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

def generar_respuesta_codigo(consulta: str, contexto: str):
    """Genera una respuesta técnica centrada en Racket con lógica de reintentos."""
    if client is None:
        raise HTTPException(status_code=500, detail="Cliente Gemini no inicializado.")
        
    prompt = f"""
    Tu rol: Eres un profesor universitario experto en **Interpretadores y Lenguajes de Programación**, especialista en **Racket** y la librería `eopl`.
    
    Tu misión: Responder a la duda del usuario basándote en la información técnica y los códigos fuente del CONTEXTO.

    ### ESCENARIOS DE RESPUESTA:

    1. **Si el usuario PIDE un ejemplo:** - Busca en el CONTEXTO el código con mejor similitud. 
       - Si el CONTEXTO contiene un ejemplo relevante: Muéstralo y explica su funcionamiento (gramática, ambientes, etc.) paso a paso.
       - Si el CONTEXTO NO contiene un ejemplo que satisfaga la petición: Indica profesionalmente: "No dispongo de un ejemplo de código específico en mis registros para explicar ese concepto."

    2. **Si el usuario ENVÍA código en su pregunta para que lo expliques:**
       - Explica el código del usuario usando tus conocimientos expertos en Racket/eopl.
       - **Búsqueda de Similitud:** Si en el CONTEXTO encuentras un código similar o relacionado, menciónalo como una referencia adicional diciendo: "He encontrado este ejemplo relacionado en los archivos del curso que podría interesarte:" y muestra el fragmento del contexto.

    3. **Si es una duda teórica sobre el código del contexto:**
       - Responde detalladamente basándote en la explicación técnica del fragmento recuperado.

    ### REGLAS TÉCNICAS INELUDIBLES:
    - **Sintaxis:** Explica el cod, su funcionalidad.
    - **Formato:** Todo código debe ir en bloques Markdown con sintaxis `racket`.
    - **Fidelidad:** No inventes funcionalidades que no existan en los fragmentos del CONTEXTO a menos que sea para explicar el código que el usuario envió.

    CONTEXTO RECUPERADO (Explicación + Código Fuente):
    {contexto}

    PREGUNTA DEL USUARIO:
    {consulta}
    """

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
                time.sleep(2**retry_count)
            else:
                raise HTTPException(status_code=500, detail=f"Error Gemini: {e}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error inesperado: {e}")

    return respuesta_final

def generar_respuesta_hibrida(consulta: str, contexto: str):
    """Genera respuestas que pueden contener teoría (PDF/Video) o práctica (Código)."""
    if client is None:
        raise HTTPException(status_code=500, detail="Cliente Gemini no inicializado.")

    prompt = f"""
    Tu rol: Eres un profesor universitario experto en **Fundamentos de Interpretación y Compilación**. Dominas tanto la teoría como la implementación práctica en **Racket (eopl)**.

    Tu tarea: Responder a la duda del usuario usando exclusivamente el CONTEXTO proporcionado.

    ### REGLAS SEGÚN EL TIPO DE INFORMACIÓN EN EL CONTEXTO:
    1. **Si hay fragmentos de CÓDIGO:** Explica la implementación, gramática o sintaxis. Si el usuario pide un ejemplo y el contexto lo tiene, muéstralo en un bloque `racket`,  Todo código debe ir en bloques Markdown con sintaxis `racket`.
    2. **Si hay teoría (PDF/Video):** Úsala para dar definiciones formales y conceptos.
    3. **Si el usuario envía su propio código:** Explícalo usando tus conocimientos y compáralo con los ejemplos del contexto si existen similitudes.
    4. **Si falta información:** Si no hay un ejemplo específico o la teoría no cubre la duda, di: "La información disponible en el material de clase no cubre este punto específico".

    CONTEXTO:
    {contexto}

    PREGUNTA: {consulta}
    """

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
                time.sleep(2**retry_count)
            else:
                raise HTTPException(status_code=500, detail=f"Error Gemini: {e}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error inesperado: {e}")

    return respuesta_final
# ==========================================================
# 3. FUNCIÓN AUXILIAR PARA LA EJECUCIÓN DE RAG
# ==========================================================

async def _ejecutar_rag(consulta: str, dataframe: pd.DataFrame, top_n: int):
    """
    Función interna que encapsula la lógica de RAG para un DataFrame específico.
    """
    
    # 1. Recuperación de documentos
    try:
        # Ahora pasamos el DataFrame específico (todo, pdf o video)
        documentos_relevantes = encontrar_documento_relevante(consulta, dataframe, MODEL_ID, top_n=top_n)
    except Exception as e:
        # Captura excepciones para formatearlas correctamente en el retorno de la API
        detail = e.detail if isinstance(e, HTTPException) else str(e)
        raise HTTPException(status_code=500, detail=f"Error en la fase de Recuperación de Documentos (Embeddings): {detail}")

    # 2. Concatenar el contexto
    contexto_combinado = "\n\n--- FUENTE ADICIONAL ---\n\n".join([
        f"Fuente: {doc['titulo']} (Similitud: {doc['similitud']:.4f})\nContenido: {doc['documento']}" 
        for doc in documentos_relevantes
    ])

    # 3. Generación de la respuesta
    if not documentos_relevantes:
        respuesta_final = f"No se encontró información relevante para su pregunta ('{consulta}') en la fuente de datos seleccionada."
        documentos_usados = []
    else:
        try:
            respuesta_final = generar_respuesta(consulta, contexto_combinado)
        except Exception as e:
            raise e # Propaga el HTTPException de generar_respuesta
        
        documentos_usados = [
            {"titulo": doc['titulo'], "metadata": doc['metadata'], "similitud": f"{doc['similitud']:.4f}"} 
            for doc in documentos_relevantes
        ]
        
    # 4. Retornar el resultado estructurado
    return {
        "pregunta": consulta,
        "respuesta": respuesta_final,
        "documentos_usados": documentos_usados
    }

# ==========================================================
# 4. ENDPOINTS DE LA API (Multifuente)
# ==========================================================

# >>> ESTE REEMPLAZA A SU ENDPOINT ORIGINAL /rag/responder <<<
@app.post("/rag/responder/todo")
@app.post("/rag/responder/todo")
async def responder_pregunta_todo(data: Consulta):
    """Endpoint que busca en TODO el corpus y maneja inteligentemente los códigos."""
    if docs_df_todo is None or docs_df_todo.empty:
        raise HTTPException(status_code=500, detail="Corpus no cargado.")

    # 1. Recuperación
    documentos = encontrar_documento_relevante(data.pregunta, docs_df_todo, MODEL_ID, top_n=data.top_n)

    # 2. Construcción de contexto inteligente
    contexto_lista = []
    for doc in documentos:
        fuente = doc['metadata'].get('FUENTE', 'DESCONOCIDA')
        contenido = doc['documento']
        
        # Si es código, extraemos el snippet de la metadata
        if fuente == 'CODIGO' and 'codigo' in doc['metadata']:
            snippet = doc['metadata']['codigo']
            item = f"--- FUENTE: CÓDIGO ({doc['titulo']}) ---\nEXPLICACIÓN: {contenido}\nCODE_START\n{snippet}\nCODE_END"
        else:
            item = f"--- FUENTE: {fuente} ({doc['titulo']}) ---\nCONTENIDO: {contenido}"
        
        contexto_lista.append(item)

    contexto_final = "\n\n".join(contexto_lista)

    # 3. Generación con la nueva lógica híbrida
    try:
        respuesta = generar_respuesta_hibrida(data.pregunta, contexto_final)
    except Exception as e:
        raise e

    return {
        "pregunta": data.pregunta,
        "respuesta": respuesta,
        "documentos_usados": [
            {
                "titulo": d['titulo'], 
                "fuente": d['metadata'].get('FUENTE'), 
                "similitud": f"{d['similitud']:.4f}"
            } for d in documentos
        ]
    }
# async def responder_pregunta_todo(data: Consulta):
#     """
#     Endpoint principal para obtener una respuesta RAG, buscando en **TODO** el corpus.
#     """
#     if docs_df_todo is None or docs_df_todo.empty:
#         raise HTTPException(status_code=500, detail="El corpus total de embeddings no está cargado o está vacío.")

#     return await _ejecutar_rag(data.pregunta, docs_df_todo, data.top_n)


@app.post("/rag/responder/pdf")
async def responder_pregunta_pdf(data: Consulta):
    """
    Endpoint para obtener una respuesta RAG, buscando **SOLO** en documentos con FUENTE: 'PDF'.
    """
    if docs_df_pdf is None or docs_df_pdf.empty:
        raise HTTPException(status_code=404, detail="El corpus de documentos PDF no está disponible o está vacío.")

    return await _ejecutar_rag(data.pregunta, docs_df_pdf, data.top_n)


@app.post("/rag/responder/video")
async def responder_pregunta_video(data: Consulta):
    """
    Endpoint para obtener una respuesta RAG, buscando **SOLO** en documentos con FUENTE: 'VIDEO'.
    """
    if docs_df_video is None or docs_df_video.empty:
        raise HTTPException(status_code=404, detail="El corpus de documentos VIDEO no está disponible o está vacío.")

    return await _ejecutar_rag(data.pregunta, docs_df_video, data.top_n)

@app.post("/rag/responder/codigo")
async def responder_pregunta_codigo(data: Consulta):
    """
    Endpoint para obtener explicaciones de código buscando en la metadata.
    """
    if docs_df_codigos is None or docs_df_codigos.empty:
        raise HTTPException(status_code=404, detail="El corpus de CÓDIGO no está cargado.")

    # 1. Recuperación de documentos relevantes
    try:
        documentos_relevantes = encontrar_documento_relevante(data.pregunta, docs_df_codigos, MODEL_ID, top_n=data.top_n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en recuperación: {str(e)}")

    if not documentos_relevantes:
        return {"pregunta": data.pregunta, "respuesta": "No encontré ejemplos de código relacionados.", "documentos_usados": []}

    # 2. Construcción del contexto UNIFICANDO contenido y el campo 'codigo' de la metadata
    contexto_combinado = ""
    for i, doc in enumerate(documentos_relevantes):
        # Extraemos el código real desde la metadata según tu ejemplo
        codigo_fuente = doc['metadata'].get('codigo', 'Código no disponible')
        explicacion_texto = doc['documento'] # Este es el campo 'contenido' del JSON
        
        contexto_combinado += f"\n--- EJEMPLO {i+1}: {doc['titulo']} ---\n"
        contexto_combinado += f"EXPLICACIÓN TÉCNICA: {explicacion_texto}\n"
        contexto_combinado += f"CÓDIGO FUENTE RACKET:\n{codigo_fuente}\n"

    # 3. Generación de la respuesta con el prompt especializado
    try:
        respuesta_final = generar_respuesta_codigo(data.pregunta, contexto_combinado)
    except Exception as e:
        raise e

    return {
        "pregunta": data.pregunta,
        "respuesta": respuesta_final,
        "documentos_usados": [
            {"titulo": doc['titulo'], "similitud": f"{doc['similitud']:.4f}", "archivo": doc['metadata'].get('NOMBRE_DOCUMENTO')} 
            for doc in documentos_relevantes
        ]
    }