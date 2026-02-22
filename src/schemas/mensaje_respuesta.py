# src/schemas/mensaje_respuesta.py

from pydantic import BaseModel
from typing import Optional


class MensajeRespuestaBase(BaseModel):
    respuesta: str


class MensajeRespuestaCreate(MensajeRespuestaBase):
    mensaje_id: int
    material_id: Optional[int] = None


class MensajeRespuestaUpdate(BaseModel):
    respuesta: Optional[str] = None
    material_id: Optional[int] = None


class MensajeRespuestaResponse(MensajeRespuestaBase):
    mensaje_id: int
    material_id: Optional[int] = None

    class Config:
        from_attributes = True