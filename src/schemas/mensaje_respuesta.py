from pydantic import BaseModel
from typing import Optional

class MensajeRespuestaBase(BaseModel):
    respuesta: str
    calificacion: Optional[int] = None

class MensajeRespuestaCreate(MensajeRespuestaBase):
    mensaje_id: int

class MensajeRespuestaUpdate(BaseModel):
    respuesta: Optional[str] = None
    calificacion: Optional[int] = None

class MensajeRespuestaResponse(MensajeRespuestaBase):
    mensaje_id: int
    class Config:
        from_attributes = True