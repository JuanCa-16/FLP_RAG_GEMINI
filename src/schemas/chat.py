# src/schemas/chat.py

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class ChatBase(BaseModel):
    titulo: Optional[str] = None


class ChatCreate(ChatBase):
    pass


class ChatUpdate(BaseModel):
    titulo: Optional[str] = None


class ChatResponse(ChatBase):
    id: int
    usuario: str
    fecha_creacion: datetime
    fecha_actualizacion: datetime

    class Config:
        from_attributes = True


class ChatConMensajes(ChatResponse):
    """Chat con contador de mensajes"""
    total_mensajes: int = 0