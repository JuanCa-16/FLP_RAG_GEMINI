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
from src.routers.usuarios import router as usuarios_router
from src.routers.material_estudio import router as material_router
from src.routers.biblioteca import router as biblioteca_router
from src.routers.chat import router as chat_router
from src.routers.mensaje import router as mensaje_router
from src.routers.auth import router as auth_router
from src.routers.auth import get_current_user
from fastapi import Depends

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

docs_df_todo = None
docs_df_pdf = None
docs_df_video = None
docs_df_codigos = None
docs_df_git= None

TOP_N_CONFIG = {
    'PDF': 2,
    'VIDEO': 2,
    'CODIGO': 2,
    'GIT': 2,
    'ALL': 2
}

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
    docs_df_git = docs_df_todo[docs_df_todo['FUENTE'] == 'GIT'].copy()

    # Log de conteo para verificación
    print(f"  - Total de documentos cargados: {len(docs_df_todo)}")
    print(f"  - Documentos PDF disponibles: {len(docs_df_pdf)}")
    print(f"  - Documentos VIDEO disponibles: {len(docs_df_video)}")
    print(f"  - Documentos GIT disponibles: {len(docs_df_git)}")

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

app.include_router(usuarios_router, prefix="/usuarios", tags=["Usuarios"])
app.include_router(material_router, prefix="/material", tags=["Material Estudio"], dependencies=[Depends(get_current_user)])
app.include_router(biblioteca_router, prefix="/biblioteca", tags=["Bilbioteca de Contenidos"], dependencies=[Depends(get_current_user)])
app.include_router(chat_router, prefix="/chat", tags=["Chats"], dependencies=[Depends(get_current_user)])
app.include_router(mensaje_router, prefix="/mensaje", tags=["Mensajes"], dependencies=[Depends(get_current_user)])
app.include_router(auth_router, prefix="/auth", tags=["Autenticación"])

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
    print('FUENTE', fuente,'CONTEXTO', contexto, 'CONSULTA:', consulta)
    """Genera una respuesta usando el modelo generativo y el contexto recuperado."""
    if client is None:
        raise HTTPException(status_code=500, detail="El cliente Gemini no se inicializó correctamente.")
    
    if (fuente == 'CÓDIGO' or fuente == 'CODIGO'  or fuente == 'CODE'):
       prompt = f"""
        Eres un profesor universitario experto en Racket y la librería eopl. Tu objetivo es SIEMPRE dar una respuesta técnica útil al estudiante.

        PRIORIDAD DE FUENTES:
        1. **PRIMERO:** Busca la respuesta en el CONTEXTO proporcionado
        2. **SEGUNDO:** Si el CONTEXTO no contiene la información exacta pero tiene ejemplos relacionados, ÚSALOS como base y adapta
        3. **TERCERO:** Si el CONTEXTO está vacío o no es relevante, usa tu conocimiento de Racket/eopl

        REGLAS DE ETIQUETADO:
        - Si usas código/explicación del CONTEXTO: No agregues etiquetas
        - Si generas código propio porque el CONTEXTO no es útil: Agrega `> **Nota:** Ejemplo generado por IA (no está en el material de clase).`
        - Si adaptas código del CONTEXTO: Agrega `> **Nota:** Adaptado de ejemplos del material de clase.`

        ESTRUCTURA DE RESPUESTA OBLIGATORIA:
        1. **Análisis breve** (1-2 frases): Qué hace el código o qué resuelve
        2. **Bloque de código** Un bloque markdown con sintaxis ```racket```
        3. **Salida esperada** en bloque de texto plano
        4. **Explicación pedagógica** (2-4 frases): Conceptos clave (recursión, listas, car/cdr, ambientes, etc.)

        CASOS ESPECIALES:
        - Si el usuario envía código con errores: explica el error brevemente, muestra la versión corregida
        - Si pide sintaxis de Racket: usa tu conocimiento (no necesitas CONTEXTO para sintaxis básica)
        - Si no está relacionado con programación/Racket: responde "Tema fuera del alcance del curso"

        CONTEXTO DISPONIBLE:
        {contexto}

        PREGUNTA DEL ESTUDIANTE:
        {consulta}

        FUENTE DEL CONTEXTO.
        {fuente}

        IMPORTANTE: Si el CONTEXTO contiene AL MENOS UN ejemplo relacionado, ÚSALO como base. No digas "no tengo información" si hay código similar en el CONTEXTO.
        """
    elif fuente in ("PDF", "VIDEO"):
          prompt = f"""
        Eres un profesor universitario experto en Fundamentos de Interpretación y Compilación (Racket/eopl).

        REGLA DE ORO: Responde ÚNICAMENTE con información del CONTEXTO. NO inventes, NO supongas, NO agregues información externa.

        INSTRUCCIONES:
        1. Lee TODO el CONTEXTO completo antes de responder
        2. Si el CONTEXTO contiene la respuesta (completa o parcial): Úsala para redactar una explicación clara
        3. Si el CONTEXTO NO contiene información relevante: Responde EXACTAMENTE: "El material de clase no cubre este tema específicamente."

        FORMATO DE RESPUESTA (cuando hay información en el CONTEXTO):
        - Redacta en texto claro y directo
        - Usa **negritas** para conceptos clave
        - Frases cortas (3-8 frases en total)
        - Si el CONTEXTO tiene fragmentos de código pequeños: inclúyelos en un bloque markdown con sintaxis ```racket```
        - Tono profesoral pero accesible

        PROHIBIDO:
        - Inventar definiciones que no están en el CONTEXTO
        - Agregar ejemplos de código no presentes en el CONTEXTO
        - Suponer información adicional
        - Expandir más allá de lo que dice el CONTEXTO

        IMPORTANTE: Este material es teórico/conceptual. Si la pregunta requiere ejemplos de código extensos y el CONTEXTO no los tiene, responde: "El material teórico no incluye ejemplos de código para esto."

        CONTEXTO:
        {contexto}

        PREGUNTA:
        {consulta}

        FUENTE DEL CONTEXTO.
        {fuente}
        """
    elif fuente == "GIT":
       prompt = f"""
    Eres un profesor universitario experto en Racket, eopl, y Fundamentos de Lenguajes de Programación.

    TIPO DE MATERIAL: Este contenido proviene de las notas de clase oficiales (GitHub) y contiene una mezcla de:
    - Explicaciones teóricas y conceptuales
    - Ejemplos de código Racket/Scheme
    - Ejercicios y aplicaciones prácticas

    MANEJO DE FÓRMULAS MATEMÁTICAS:
    - Si encuentras notación LaTeX corrupta (ej: $$ \\begin{{align}} ... $$), INTENTA reconstruir la fórmula en notación clara
    - Si la fórmula es simple: reescríbela en texto plano legible (ej: "n ∈ ℕ → n+1 ∈ ℕ")
    - Si la fórmula es compleja o no está clara: OMÍTELA y enfócate en explicar el concepto en palabras
    - NUNCA incluyas código LaTeX roto en tu respuesta

    FORMATO MARKDOWN PARA CONCEPTOS TÉCNICOS:
    - Usa `backticks` para nombres de funciones: `define`, `lambda`, `car`, `cdr`
    - Usa `backticks` para palabras clave de Racket: `let`, `cond`, `if`, `cons`
    - Usa `backticks` para nombres de variables: `x`, `lst`, `acc`
    - Usa `backticks` para tipos de datos: `list`, `number`, `boolean`
    - Usa `backticks` para símbolos especiales: `'()`, `#t`, `#f`
    - Usa **negritas** solo para conceptos teóricos abstractos: **recursión**, **clausura léxica**, **evaluación perezosa**

    ESTRATEGIA DE RESPUESTA:

    1. **Analiza la pregunta:**
       - ¿Es conceptual/teórica? → Enfócate en explicaciones del CONTEXTO
       - ¿Requiere código/ejemplos? → Usa los ejemplos del CONTEXTO
       - ¿Es mixta (teoría + práctica)? → Combina ambos del CONTEXTO

    2. **REGLAS ESTRICTAS:**
       - **PRIORIDAD ABSOLUTA:** Usa el CONTEXTO como fuente principal
       - Si el CONTEXTO tiene la explicación: úsala directamente (reformula si es necesario)
       - Si el CONTEXTO tiene código de ejemplo: úsalo tal cual o adáptalo mínimamente
       - Si el CONTEXTO no es suficiente pero es parcialmente relevante: combina lo que hay con conocimiento de Racket
       - Si el CONTEXTO no es relevante en absoluto: usa tu conocimiento pero marca claramente

    3. **FORMATO DE RESPUESTA:**

    **Para preguntas conceptuales:**
    - Explicación clara en 3-6 frases
    - Usa `backticks` para todos los términos técnicos y código
    - Usa **negritas** solo para conceptos teóricos abstractos
    - Si el CONTEXTO tiene código ilustrativo pequeño, inclúyelo en ```racket```

    **Para preguntas de código:**
    - Breve explicación (1-2 frases) del propósito
    - Bloque ```racket``` con el código
    - Salida esperada o resultado
    - Explicación de conceptos clave aplicados (2-3 frases) usando `backticks` para términos técnicos

    **Para preguntas mixtas:**
    - Explicación conceptual primero (2-3 frases) con `backticks` para términos técnicos
    - Código de ejemplo en ```racket```
    - Salida/resultado
    - Conexión teoría-práctica (1-2 frases)

    4. **ETIQUETADO:**
    - Si usas contenido directo del CONTEXTO: sin etiqueta
    - Si adaptas significativamente ejemplos del CONTEXTO: `> **Nota:** Adaptado del material de clase.`
    - Si generas código nuevo porque el CONTEXTO no lo tiene: `> **Nota:** Ejemplo generado por IA (no está en las notas de clase).`
    - Si combinas CONTEXTO + conocimiento propio: `> **Nota:** Basado en material de clase con explicación extendida.`

    5. **CASOS ESPECIALES:**
    - **Código con errores:** Identifica el error, explícalo brevemente, muestra versión corregida
    - **Sintaxis básica de Racket:** Usa tu conocimiento (`define`, `lambda`, `cond`, etc.)
    - **Preguntas fuera de alcance:** "Este tema no está cubierto en las notas de clase."
    - **Múltiples enfoques en el CONTEXTO:** Muestra el más relevante y menciona que hay alternativas

    6. **TONO Y ESTILO:**
    - Profesoral pero accesible
    - Didáctico: explica el "por qué", no solo el "cómo"
    - Usa terminología técnica correcta con formato de código inline
    - Ejemplos concretos siempre que sea posible

    EJEMPLOS DE USO CORRECTO DE FORMATO:

    ✅ CORRECTO:
    "La función `car` extrae el primer elemento de una lista, mientras que `cdr` retorna el resto. 
    Juntas permiten implementar **recursión** sobre listas."

    ❌ INCORRECTO:
    "La función **car** extrae el primer elemento de una lista, mientras que **cdr** retorna el resto. 
    Juntas permiten implementar recursión sobre listas."

    ✅ CORRECTO:
    "En Racket, `define` declara una variable global. Para funciones anónimas usamos `lambda`."

    ❌ INCORRECTO:
    "En Racket, **define** declara una variable global. Para funciones anónimas usamos **lambda**."

    CONTEXTO DISPONIBLE (Notas de clase GitHub):
    {contexto}

    PREGUNTA DEL ESTUDIANTE:
    {consulta}

    FUENTE DEL CONTEXTO:
    {fuente}

    IMPORTANTE: 
    - Las notas de clase son tu fuente PRIMARIA - úsalas exhaustivamente
    - Si el CONTEXTO tiene CUALQUIER información relacionada, inclúyela en tu respuesta
    - Mantén coherencia con la nomenclatura y estilo de las notas de clase
    - Siempre completa la respuesta, no dejes conceptos a medias
    - Usa `backticks` para TODOS los términos técnicos de programación
    """
    
    else: 
        # ALL
       prompt = f"""
        Eres un profesor universitario experto en Racket, eopl, y Fundamentos de Compilación.

        CONTEXTO MIXTO: Este CONTEXTO puede contener material teórico (PDF/VIDEO), notas de clase (GIT), y ejemplos de código.

        ESTRATEGIA DE RESPUESTA:
        1. **Identifica el tipo de pregunta:**
           - ¿Es sobre conceptos teóricos? → Usa material PDF/VIDEO/GIT del CONTEXTO (NO inventes)
           - ¿Requiere código/ejemplos? → Usa ejemplos de código del CONTEXTO (puedes generar si no hay)

        2. **Para CONCEPTOS TEÓRICOS:**
           - Usa SOLO el CONTEXTO (material PDF/VIDEO/GIT)
           - Si el CONTEXTO no tiene la información: "El material de clase no cubre este tema específicamente."
           - Redacta explicaciones claras con **conceptos clave** en negritas
           - NO inventes teoría

        3. **Para EJEMPLOS DE CÓDIGO:**
           - PRIORIDAD 1: Usa ejemplos del CONTEXTO si existen (GIT o CÓDIGO)
           - PRIORIDAD 2: Si no hay ejemplos relevantes, genera uno usando tu conocimiento de Racket
           - Siempre marca código generado: `> **Nota:** Ejemplo generado por IA (no está en el material de clase).`
           - Incluye siempre la salida esperada

        4. **Para PREGUNTAS MIXTAS (teoría + código):**
           - Teoría: SOLO del CONTEXTO
           - Código: Del CONTEXTO o generado si es necesario

        FORMATO DE RESPUESTA:
        - **Explicación teórica:** Texto claro, frases cortas, conceptos en **negritas**
        - **Código:** Bloque ```racket``` + salida esperada
        - Longitud: 3-8 frases + código si aplica

        MANEJO DE FÓRMULAS MATEMÁTICAS:
        - Si encuentras notación LaTeX corrupta (ej: $$ \begin{{align}} ... $$), INTENTA reconstruir la fórmula en notación clara
        - Si la fórmula es simple: reescríbela en texto plano legible (ej: "n ∈ ℕ → n+1 ∈ ℕ")
        - Si la fórmula es compleja o no está clara: OMÍTELA y enfócate en explicar el concepto en palabras
        - NUNCA incluyas código LaTeX roto en tu respuesta

        REGLAS ESPECÍFICAS:
        - Si el usuario envía código: explícalo usando conceptos del CONTEXTO
        - Si hay código con errores en el CONTEXTO: muestra la versión corregida
        - Sintaxis básica de Racket: puedes usar tu conocimiento
        - Siempre completa la respuesta, no dejes ideas a medias

        CONTEXTO DISPONIBLE:
        {contexto}

        PREGUNTA:
        {consulta}

        FUENTE DEL CONTEXTO:
        {fuente}

        RECUERDA: 
        - Material teórico → SOLO CONTEXTO, no inventes
        - Ejemplos de código → Prioriza CONTEXTO, genera si es necesario marcándolo
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
                    'temperature': 0.1,
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
        top_n_usar = TOP_N_CONFIG.get(fuente, 2) 
        #antes:  top_n=data.top_n
        documentos = encontrar_documento_relevante(pregunta_optimizada, dataframe, MODEL_ID, top_n= top_n_usar)
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

    return await _ejecutar_rag(data, docs_df_codigos, 'CODIGO')

@app.post("/rag/responder/git")
async def responder_pregunta_git(data: Consulta):

    if docs_df_git is None or docs_df_git.empty:
        raise HTTPException(status_code=404, detail="El corpus de GIT no está cargado.")

    return await _ejecutar_rag(data, docs_df_git, 'GIT')

# uvicorn app:app --reload