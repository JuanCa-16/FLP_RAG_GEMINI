from pydantic import BaseModel,Field
from typing import Optional

class MensajeRespuestaBase(BaseModel):
    respuesta: str
    calificacion: Optional[int] = Field(None, ge=1, le=5)

class MensajeRespuestaCreate(MensajeRespuestaBase):
    mensaje_id: int

class MensajeRespuestaUpdate(BaseModel):
    respuesta: Optional[str] = None
    calificacion: Optional[int] = Field(None, ge=1, le=5)

class MensajeRespuestaResponse(MensajeRespuestaBase):
    mensaje_id: int
    class Config:
        from_attributes = True
