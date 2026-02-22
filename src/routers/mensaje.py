# src/routes/mensaje.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional

from src.database.database import SessionLocal
from src.models.mensaje import Mensaje
from src.models.mensaje_pregunta import MensajePregunta
from src.models.mensaje_respuesta import MensajeRespuesta
from src.models.chat import Chat
from src.schemas.mensaje import MensajeCreate, MensajeResponse
from src.schemas.mensaje_pregunta import MensajePreguntaCreate
from src.schemas.mensaje_respuesta import MensajeRespuestaCreate

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 🔹 Crear mensaje con pregunta (completo)
@router.post("/pregunta")
def crear_mensaje_pregunta(
    chat_id: int,
    rol: str,
    tipo: str,
    pregunta: str,
    pregunta_embedding: Optional[List[float]] = None,
    db: Session = Depends(get_db)
):
    # Verificar que el chat existe
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat no encontrado")
    
    # 1. Crear mensaje
    nuevo_mensaje = Mensaje(
        chat_id=chat_id,
        rol=rol,
        tipo=tipo
    )
    db.add(nuevo_mensaje)
    db.flush()  # Para obtener el ID
    
    # 2. Crear pregunta asociada
    nueva_pregunta = MensajePregunta(
        mensaje_id=nuevo_mensaje.id,
        pregunta=pregunta,
        pregunta_embedding=pregunta_embedding
    )
    db.add(nueva_pregunta)
    
    db.commit()
    db.refresh(nuevo_mensaje)
    
    return {
        "mensaje_id": nuevo_mensaje.id,
        "chat_id": chat_id,
        "pregunta": pregunta,
        "fecha_mensaje": nuevo_mensaje.fecha_mensaje
    }


# 🔹 Crear mensaje con respuesta (completo)
@router.post("/respuesta")
def crear_mensaje_respuesta(
    chat_id: int,
    rol: str,
    tipo: str,
    respuesta: str,
    material_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    # Verificar que el chat existe
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat no encontrado")
    
    # 1. Crear mensaje
    nuevo_mensaje = Mensaje(
        chat_id=chat_id,
        rol=rol,
        tipo=tipo
    )
    db.add(nuevo_mensaje)
    db.flush()
    
    # 2. Crear respuesta asociada
    nueva_respuesta = MensajeRespuesta(
        mensaje_id=nuevo_mensaje.id,
        respuesta=respuesta,
        material_id=material_id
    )
    db.add(nueva_respuesta)
    
    db.commit()
    db.refresh(nuevo_mensaje)
    
    return {
        "mensaje_id": nuevo_mensaje.id,
        "chat_id": chat_id,
        "respuesta": respuesta,
        "material_id": material_id,
        "fecha_mensaje": nuevo_mensaje.fecha_mensaje
    }


# 🔹 Obtener historial completo de un chat
@router.get("/chat/{chat_id}/historial")
def obtener_historial_chat(chat_id: int, db: Session = Depends(get_db)):
    # Verificar que el chat existe
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat no encontrado")
    
    # Obtener todos los mensajes del chat
    mensajes = (
        db.query(Mensaje)
        .filter(Mensaje.chat_id == chat_id)
        .order_by(Mensaje.fecha_mensaje.asc())
        .all()
    )
    
    historial = []
    for mensaje in mensajes:
        item = {
            "mensaje_id": mensaje.id,
            "rol": mensaje.rol,
            "tipo": mensaje.tipo,
            "fecha_mensaje": mensaje.fecha_mensaje,
            "contenido": None,
            "material_id": None
        }
        
        # Buscar si es pregunta
        pregunta = db.query(MensajePregunta).filter(
            MensajePregunta.mensaje_id == mensaje.id
        ).first()
        
        if pregunta:
            item["contenido"] = pregunta.pregunta
        else:
            # Buscar si es respuesta
            respuesta = db.query(MensajeRespuesta).filter(
                MensajeRespuesta.mensaje_id == mensaje.id
            ).first()
            
            if respuesta:
                item["contenido"] = respuesta.respuesta
                item["material_id"] = respuesta.material_id
        
        historial.append(item)
    
    return {
        "chat_id": chat_id,
        "titulo": chat.titulo,
        "total_mensajes": len(historial),
        "historial": historial
    }


# 🔹 Obtener un mensaje específico
@router.get("/{mensaje_id}")
def obtener_mensaje(mensaje_id: int, db: Session = Depends(get_db)):
    mensaje = db.query(Mensaje).filter(Mensaje.id == mensaje_id).first()
    
    if not mensaje:
        raise HTTPException(status_code=404, detail="Mensaje no encontrado")
    
    resultado = {
        "mensaje_id": mensaje.id,
        "chat_id": mensaje.chat_id,
        "rol": mensaje.rol,
        "tipo": mensaje.tipo,
        "fecha_mensaje": mensaje.fecha_mensaje,
        "contenido": None,
        "material_id": None
    }
    
    # Buscar contenido
    pregunta = db.query(MensajePregunta).filter(
        MensajePregunta.mensaje_id == mensaje.id
    ).first()
    
    if pregunta:
        resultado["contenido"] = pregunta.pregunta
    else:
        respuesta = db.query(MensajeRespuesta).filter(
            MensajeRespuesta.mensaje_id == mensaje.id
        ).first()
        
        if respuesta:
            resultado["contenido"] = respuesta.respuesta
            resultado["material_id"] = respuesta.material_id
    
    return resultado


# 🔹 Eliminar mensaje
@router.delete("/{mensaje_id}")
def eliminar_mensaje(mensaje_id: int, db: Session = Depends(get_db)):
    mensaje = db.query(Mensaje).filter(Mensaje.id == mensaje_id).first()
    
    if not mensaje:
        raise HTTPException(status_code=404, detail="Mensaje no encontrado")
    
    db.delete(mensaje)
    db.commit()
    
    return {"message": "Mensaje eliminado correctamente"}


# 🔹 Actualizar embedding de una pregunta
@router.put("/pregunta/{mensaje_id}/embedding")
def actualizar_embedding_pregunta(
    mensaje_id: int,
    embedding: List[float],
    db: Session = Depends(get_db)
):
    pregunta = db.query(MensajePregunta).filter(
        MensajePregunta.mensaje_id == mensaje_id
    ).first()
    
    if not pregunta:
        raise HTTPException(status_code=404, detail="Pregunta no encontrada")
    
    if len(embedding) != 768:
        raise HTTPException(
            status_code=400,
            detail="El embedding debe tener 768 dimensiones"
        )
    
    pregunta.pregunta_embedding = embedding
    db.commit()
    
    return {"message": "Embedding actualizado correctamente"}