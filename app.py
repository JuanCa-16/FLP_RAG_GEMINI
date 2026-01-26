from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import numpy as np
from numpy.linalg import norm
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
import time
from google.genai.errors import ServerError
from pydantic import Field

# CONFIGURACIÓN y CARGA GLOBAL
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Inicialización global del cliente y carga de datos. Esto se hace una sola vez al iniciar el servidor

try:
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"Error al inicializar el cliente Gemini: {e}")
    client = None

MODEL_ID = "gemini-embedding-001"
GENERATIVE_MODEL = "gemini-2.5-flash"
RUTA_EMBEDDINGS = "src/embeddings/corpus_con_ejemplos_embeddings.jsonl"

docs_df_all = None
docs_df_pdf = None
docs_df_video = None
docs_df_codigos = None

print("✅ Cargando y filtrando embeddings existentes...")
try:
    # 1. Carga del DataFrame COMPLETO (se hace una sola vez)
    docs_df_todo = pd.read_json(RUTA_EMBEDDINGS, lines=True)
    
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
    print(f"Error: No se encontró el archivo {RUTA_EMBEDDINGS}.")
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
class Mensaje(BaseModel):
    role: str  # "user" o "model"
    content: str

class Consulta(BaseModel):
    pregunta: str
    top_n: int = 1
    historial: list[Mensaje] = Field(default_factory=list)

# FUNCIONES DE RAG

def encontrar_documento_relevante(consulta: str, dataframe: pd.DataFrame, modelo: str, top_n: int = 3):
    """Encuentra los top N documentos más relevantes para la consulta dada."""
    if client is None:
        raise HTTPException(status_code=500, detail="El cliente Gemini no se inicializó correctamente.")
    
    if dataframe is None or dataframe.empty:
        raise HTTPException(status_code=404, detail="El corpus de documentos seleccionado está vacío.")

    # Generamos el embedding para la consulta con RETRIEVAL_QUERY
    embedding_consulta = client.models.embed_content(
        model=modelo,
        contents=consulta,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY",output_dimensionality=768)
    )

    [embedding_reducido] = embedding_consulta.embeddings

    embedding_values_np = np.array(embedding_reducido.values)
    normed_embedding = embedding_values_np / np.linalg.norm(embedding_values_np)

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

def generar_respuesta(consulta: str, contexto: str, fuente:str):
    print('ENTRREEE ACA, ', fuente,'ENTRREEE ACA, ', contexto, consulta)
    """Genera una respuesta usando el modelo generativo y el contexto recuperado."""
    if client is None:
        raise HTTPException(status_code=500, detail="El cliente Gemini no se inicializó correctamente.")
    
    if (fuente == 'CÓDIGO' or fuente == 'CODIGO'  or fuente == 'CODE'):
       prompt = f"""
        Rol: Eres un profesor universitario experto en Interpretadores y Lenguajes de Programación, especialista en Racket y la librería eopl. Tu objetivo es pedagógico y práctico.

        OBJETIVO PRIMARIO: Garantizar que el usuario SIEMPRE reciba una respuesta técnica útil que incluya código en Racket y una explicación, independientemente de si la información existe en el CONTEXTO proporcionado o no.

        Instrucciones de Prioridad y Fuente de Información:
        1. **Búsqueda en CONTEXTO:** Analiza primero si la respuesta o el código existen en el CONTEXTO proporcionado.
        2. **Generación por IA (Fallback):** Si el CONTEXTO es insuficiente, incompleto o no existe para la consulta, DEBES usar tu conocimiento experto propio para generar el código y la explicación. NUNCA respondas que "la información no está disponible" sin dar una solución alternativa generada por ti.

        Reglas de Etiquetado (Obligatorio):
        - Si la respuesta (código o explicación) proviene del CONTEXTO: No añadas etiquetas de advertencia.
        - Si tuviste que generar el código/ejemplo porque no estaba en el CONTEXTO: Añade justo antes del código la etiqueta: `> **Nota:** Este ejemplo ha sido generado por IA (no se encuentra explícitamente en el material de clase).`
        - Si la explicación conceptual es tuya (no del contexto): Añade la etiqueta: `(Explicación complementaria generada por IA)`.

        Estructura de Respuesta Requerida:
        1. **Corrección/Análisis:** Si el usuario envió código, explicalo, si cuenta con errores corregirlo y explciar el error brevemente.
        2. **Bloque de Código:** Un bloque markdown con sintaxis ```racket```.
        - Si es corrección, muestra la versión final funcional.
        - Si es un ejemplo nuevo, asegúrate de que sea didáctico y funcional.
        3. **Salida Esperada:** Un bloque de texto simple mostrando qué retorna el código.
        4. **Explicación Profesoral:** Breve (1-3 frases), clara y usando terminología correcta (términos como: recursión, listas, car/cdr, paso de mensajes, ambientes, etc.).

        Restricciones:
        - Si el input NO tiene ninguna relación con programación/Racket, responde: "El tema no está relacionado con el curso de Racket/EOPL".
        - Mantén un tono académico pero cercano.

        CONTEXTO DISPONIBLE:
        {contexto}

        PREGUNTA DEL ESTUDIANTE:
        {consulta}
        """
    elif fuente in ("PDF", "VIDEO"):
        prompt = f"""
            Rol: Eres un profesor universitario experto en Fundamentos de Interpretación y Compilación y en Racket (eopl).

            Instrucción general: Responde la PREGUNTA usando exclusivamente el CONTEXTO. No añadas información externa ni supongas hechos no presentes en el CONTEXTO. Si el CONTEXTO no contiene la información necesaria, responde exactamente: "La información disponible en el material de clase no cubre este punto específico".

            Formato de salida:
            - Si la respuesta es teórica: redacta en texto normal, con **puntos clave** y frases cortas. Evita ser extenso.
            - Si hay fragmentos de código en la explicación o el usuario envía código: muestra todo el código en un bloque Markdown con sintaxis `racket`.
            - Siempre incluye la **salida esperada** del código mostrado.
            - Si el usuario envía su propio código: explícalo y compáralo con ejemplos del CONTEXTO si existen similitudes.
            - No dejes la respuesta a medias. Sé claro y conciso.

            Reglas específicas:
            1. **Usa solo el CONTEXTO** para fundamentar la respuesta.
            2. **No inventes** definiciones, propiedades o ejemplos que no estén en el CONTEXTO.
            3. **Código:** todo código en ```racket```; si no hay código en el CONTEXTO y la pregunta requiere ejemplo, crea un ejemplo breve y explícitalo como tal.
            4. **Si falta información:** responde exactamente: "La información disponible en el material de clase no cubre este punto específico".
            5. **Longitud:** respuesta breve (3–8 frases) más bloques de código si aplica. Nunca dejar la idea incompleta por cumplir longitud.
            6. Si el CONTEXTO contiene código y ese código presenta errores: muestra primero una breve explicación del error y luego presenta la versión corregida del código. No incluyas el código con el error; solo muestra la corrección y la salida esperada.
            7. Si la PREGUNTA solicita código y el CONTEXTO no contiene ningún ejemplo aplicable, genera un ejemplo breve y funcional en Racket que resuelva la pregunta de la forma más concisa posible. Marca explícitamente el ejemplo como: "Ejemplo generado por IA (no está en el material de clase)". Muestra el código solo en un bloque racket y, debajo, indica la salida esperada. No añadas más explicación que la estrictamente necesaria (1–3 frases) y no inventes conceptos fuera del CONTEXTO.
            8. Responde siempre la totalidad de la idea; no dejes la pregunta a medias.
            9. Usa un tono profesoral, claro y accesible; prioriza la comprensión del estudiante.

            CONTEXTO:
            {contexto}

            PREGUNTA:
            {consulta}
            """

    else: 
        # ALL
        prompt = f"""
            Rol: Eres un profesor universitario experto en Fundamentos de Interpretación y Compilación y en Racket (eopl).

            Instrucción general: Responde la PREGUNTA usando exclusivamente el CONTEXTO. No añadas información externa ni supongas hechos no presentes en el CONTEXTO. Si el CONTEXTO no contiene la información necesaria, responde exactamente: "La información disponible en el material de clase no cubre este punto específico".

            Formato de salida:
            - Si la respuesta es teórica: redacta en texto normal, con **puntos clave** y frases cortas. Evita ser extenso.
            - Si hay fragmentos de código en la explicación o el usuario envía código: muestra todo el código en un bloque Markdown con sintaxis `racket`.
            - Siempre incluye la **salida esperada** del código mostrado.
            - Si el usuario envía su propio código: explícalo y compáralo con ejemplos del CONTEXTO si existen similitudes.
            - No dejes la respuesta a medias. Sé claro y conciso.

            Reglas específicas:
            1. **Usa solo el CONTEXTO** para fundamentar la respuesta.
            2. **No inventes** definiciones, propiedades o ejemplos que no estén en el CONTEXTO.
            3. **Código:** todo código en ```racket```; si no hay código en el CONTEXTO y la pregunta requiere ejemplo, crea un ejemplo breve y explícitalo como tal.
            4. **Si falta información:** responde exactamente: "La información disponible en el material de clase no cubre este punto específico".
            5. **Longitud:** respuesta breve (3–8 frases) más bloques de código si aplica. Nunca dejar la idea incpmpleta por cumplir longitud.
            6. Si el CONTEXTO contiene código y ese código presenta errores. No incluyas el código con el error; solo muestra la corrección y la salida esperada.
            7. Si el usuario solicita en la pregunta código responde con un código.

            CONTEXTO:
            {contexto}

            PREGUNTA:
            {consulta}
            """
        
    respuesta_final = None
    max_retries = 5
    retry_count = 0

    while respuesta_final is None and retry_count < max_retries:
        try:
            respuesta = client.models.generate_content(
                model=GENERATIVE_MODEL,
                contents=prompt,
                config={
                    'temperature': 0,
                } 
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

# 3. FUNCIÓN AUXILIAR PARA LA EJECUCIÓN DE RAG
def contextualizar_pregunta(pregunta_usuario: str, historial: list[Mensaje]):
    if not historial:
        return pregunta_usuario

    chat_history_str = "\n".join([f"{m.role}: {m.content}" for m in historial])
    
    prompt = f"""
    Eres un experto en recuperación de información. Tu tarea es decidir si una "Pregunta de Usuario" necesita contexto del historial para ser entendida por un buscador.

    REGLAS:
    1. Si la pregunta es general, teórica o ya incluye sus propios sujetos (ej. "¿Cómo es la interfaz de datos recursivos?"), DEVUÉLVELA EXACTAMENTE IGUAL.
    2. Si la pregunta usa pronombres o referencias ambiguas (ej. "¿Cómo funciona eso?", "Dame otro ejemplo", "¿Quién lo creó?"), REFORMÚLALA usando el historial.
    3. NO añadas temas del historial si la pregunta del usuario ya es una consulta técnica completa por sí misma.
    4. La salida debe ser SOLO la pregunta, sin explicaciones.

    HISTORIAL:
    {chat_history_str}

    PREGUNTA DEL USUARIO: {pregunta_usuario}

    PREGUNTA PARA EL BUSCADOR:"""

    try:
        res = client.models.generate_content(
            model=GENERATIVE_MODEL, 
            contents=prompt,
            config={'temperature': 0} # Mantener 0 es vital para consistencia
        )
        return res.text.strip()
    except:
        return pregunta_usuario
    
async def _ejecutar_rag(data: Consulta, dataframe: pd.DataFrame,fuente: str):
    """
    Función interna que encapsula la lógica de RAG para un DataFrame específico.
    """
    # 1. Optimizar pregunta.
    pregunta_optimizada = contextualizar_pregunta(data.pregunta, data.historial)
    
    # 2. Recuperación de documentos
    try:
        documentos = encontrar_documento_relevante(pregunta_optimizada, dataframe, MODEL_ID, top_n=data.top_n)
    except Exception as e:
        detail = e.detail if isinstance(e, HTTPException) else str(e)
        raise HTTPException(status_code=500, detail=f"Error en la fase de Recuperación de Documentos (Embeddings): {detail}")

    # 3. Construcción del contexto inteligente
    contexto_lista = []
    for doc in documentos:
        fuente = doc['metadata'].get('FUENTE', 'DESCONOCIDA')
        contenido = doc['documento']
        
        # Si es código, extraemos el snippet de la metadata
        if (fuente == 'CÓDIGO' or fuente == 'CODIGO' or fuente == 'CODE') and 'codigo' in doc['metadata']:
            snippet = doc['metadata']['codigo']
            item = f"--- FUENTE: CÓDIGO ({doc['titulo']}) ---\nEXPLICACIÓN TÉCNICA: {contenido}\CÓDIGO FUENTE RACKET:\n{snippet}\n"
        else:
            item = f"--- FUENTE: {fuente} ({doc['titulo']}) ---\nCONTENIDO: {contenido}"
        
        contexto_lista.append(item)

    contexto_final = "\n\n".join(contexto_lista)

    # 4. Generar respuesta
    try:
        respuesta = generar_respuesta(pregunta_optimizada, contexto_final, fuente)
    except Exception as e:
        raise e
    
    return {
        "pregunta": pregunta_optimizada,
        "respuesta": respuesta,
        "documentos_usados": [
            {
                "similitud": f"{d['similitud']:.4f}",
                "metadata": d['metadata']
            } for d in documentos
        ]
    }

# 4. ENDPOINTS DE LA API 

@app.post("/rag/responder/todo")
async def responder_pregunta_todo(data: Consulta):

    if docs_df_todo is None or docs_df_todo.empty:
        raise HTTPException(status_code=500, detail="Corpus no cargado.")
    
    return await _ejecutar_rag(data, docs_df_todo, 'ALL')

@app.post("/rag/responder/pdf")
async def responder_pregunta_pdf(data: Consulta):

    if docs_df_pdf is None or docs_df_pdf.empty:
        raise HTTPException(status_code=404, detail="El corpus de documentos PDF no está disponible o está vacío.")

    return await _ejecutar_rag(data, docs_df_pdf, 'PDF')

@app.post("/rag/responder/video")
async def responder_pregunta_video(data: Consulta):

    if docs_df_video is None or docs_df_video.empty:
        raise HTTPException(status_code=404, detail="El corpus de documentos VIDEO no está disponible o está vacío.")

    return await _ejecutar_rag(data, docs_df_video, 'VIDEO')

@app.post("/rag/responder/codigo")
async def responder_pregunta_codigo(data: Consulta):

    if docs_df_codigos is None or docs_df_codigos.empty:
        raise HTTPException(status_code=404, detail="El corpus de CÓDIGO no está cargado.")

    return await _ejecutar_rag(data, docs_df_codigos, 'CÓDIGO')
