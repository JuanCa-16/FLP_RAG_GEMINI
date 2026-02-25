# src/routes/chat.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List

from src.database.database import SessionLocal
from src.models.chat import Chat
from src.models.mensaje import Mensaje
from src.schemas.chat import ChatCreate, ChatResponse, ChatUpdate, ChatConMensajes
from src.routers.auth import get_current_user

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 🔹 Crear chat
@router.post("/", response_model=ChatResponse)
def crear_chat(chat: ChatCreate,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)):

    nuevo_chat = Chat(
        usuario=current_user.usuario,
        titulo=chat.titulo
    )
    
    db.add(nuevo_chat)
    db.commit()
    db.refresh(nuevo_chat)
    
    return nuevo_chat


# 🔹 Listar todos los chats de un usuario
@router.get("/usuario/{username}", response_model=List[ChatConMensajes])
def listar_chats_usuario(username: str, db: Session = Depends(get_db)):
    chats = (
        db.query(
            Chat,
            func.count(Mensaje.id).label("total_mensajes")
        )
        .outerjoin(Mensaje, Chat.id == Mensaje.chat_id)
        .filter(Chat.usuario == username)
        .group_by(Chat.id)
        .order_by(Chat.fecha_actualizacion.desc())
        .all()
    )
    
    resultado = []
    for chat, total in chats:
        chat_dict = {
            "id": chat.id,
            "usuario": chat.usuario,
            "titulo": chat.titulo,
            "fecha_creacion": chat.fecha_creacion,
            "fecha_actualizacion": chat.fecha_actualizacion,
            "total_mensajes": total or 0
        }
        resultado.append(chat_dict)
    
    return resultado


# 🔹 Obtener chat por ID
@router.get("/{chat_id}", response_model=ChatResponse)
def obtener_chat(chat_id: int, db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat no encontrado")
    
    return chat


# 🔹 Actualizar título del chat
@router.put("/{chat_id}", response_model=ChatResponse)
def actualizar_chat(chat_id: int, chat_update: ChatUpdate, db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat no encontrado")
    
    if chat_update.titulo is not None:
        chat.titulo = chat_update.titulo
    
    db.commit()
    db.refresh(chat)
    
    return chat


# 🔹 Eliminar chat
@router.delete("/{chat_id}")
def eliminar_chat(chat_id: int, db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat no encontrado")
    
    db.delete(chat)
    db.commit()
    
    return {"message": "Chat eliminado correctamente"}


# 🔹 Obtener el chat más reciente de un usuario
@router.get("/usuario/{username}/reciente", response_model=ChatResponse)
def obtener_chat_reciente(username: str, db: Session = Depends(get_db)):
    chat = (
        db.query(Chat)
        .filter(Chat.usuario == username)
        .order_by(Chat.fecha_actualizacion.desc())
        .first()
    )
    
    if not chat:
        raise HTTPException(status_code=404, detail="No se encontraron chats para este usuario")
    
    return chat