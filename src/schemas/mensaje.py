# src/schemas/mensaje.py

from pydantic import BaseModel
from typing import Optional, Literal
from datetime import datetime


class MensajeBase(BaseModel):
    rol: Literal['assistant', 'user']
    tipo: Literal['CODIGO', 'PDF', 'VIDEO', 'GIT']


class MensajeCreate(MensajeBase):
    chat_id: int


class MensajeResponse(MensajeBase):
    id: int
    chat_id: int
    fecha_mensaje: datetime

    class Config:
        from_attributes = True