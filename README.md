# FLP_RAG_GEMINI

## Descripción principal

El sistema destaca por su motor de **recomendación personalizada de materiales educativos** para el curso de "Fundamentos
de Compilación e Interpretación de Lenguajes de Programación". Cada usuario tiene un **vector de perfil** que resume sus intereses y necesidades, construido a partir de:
- **Interacciones registradas**: Cada vez que el usuario consulta, califica o pregunta sobre un material, se almacena la interacción y se actualiza su perfil.
- **Historial de interacción (perfil implícito)**: Basado en los materiales consultados, ponderando más las acciones recientes.
- **Feedback explícito**: Calificaciones positivas y negativas sobre materiales.
- **Intención reciente**: Embeddings de las preguntas más recientes.

El perfil final es una combinación lineal de estos componentes, normalizado, y se compara con los embeddings de los materiales para recomendar los más relevantes según la similitud.

---

## Funcionalidades adicionales
- **Preguntas y respuestas (RAG)** sobre PDFs, videos, código y notas Git.
- **Chat autenticado** con historial de preguntas y respuestas.
- **Gestión de usuarios y materiales**.
- **Recomendaciones personalizadas** en base al perfil dinámico del usuario.

---

## Estructura de carpetas
- **app.py**: Aplicación principal FastAPI (servidor de la API).
- **main.py**: Script para creación y procesamiento de embeddings.
- **requirements.txt, Dockerfile**: Dependencias y despliegue.
- **data/**: Materiales del curso (PDFs, videos, código, notas Git).
- **scripts/**: Procesamiento y generación de corpus/datos.
- **src/**: Lógica principal backend:
  - **core/**: Seguridad.
  - **database/**: Modelos y conexión BD.
  - **embeddings/**: Embeddings precomputados.
  - **models/**: Modelos Pydantic/ORM y registro de interacciones.
  - **routers/**: Rutas FastAPI (auth, chat, documentos, recomendaciones, etc).
  - **schemas/**: Esquemas de validación.
  - **services/**: Lógica de negocio (recomendador).

---

## Endpoints principales

### Públicos
- **POST /rag/responder/todo**: Preguntar usando todos los materiales.
- **POST /rag/responder/pdf**: Solo PDFs.
- **POST /rag/responder/video**: Solo videos.
- **POST /rag/responder/codigo**: Solo código.
- **POST /rag/responder/git**: Solo notas Git.

### Autenticados
- **POST /rag/chat/todo**: Preguntar (todos los materiales) y guardar en historial.
- **POST /rag/chat/pdf**: Solo PDFs, con historial.
- **POST /rag/chat/video**: Solo videos, con historial.
- **POST /rag/chat/codigo**: Solo código, con historial.
- **POST /rag/chat/git**: Solo notas Git, con historial.

### Otros
- **/usuarios**: Gestión de usuarios.
- **/recomendar**: Recomendaciones y perfil de usuario.
- **/mensaje**: Mensajes y feedback.
- **/material**: CRUD de materiales.
- **/documentos**: CRUD de documentos.
- **/biblioteca**: Biblioteca de usuario.
- **/chat**: Gestión de chats.
- **/auth**: Autenticación.
