# src/schemas/mensaje_pregunta.py

from pydantic import BaseModel
from typing import Optional, List


class MensajePreguntaBase(BaseModel):
    pregunta: str


class MensajePreguntaCreate(MensajePreguntaBase):
    mensaje_id: int
    pregunta_embedding: Optional[List[float]] = None


class MensajePreguntaResponse(MensajePreguntaBase):
    mensaje_id: int

    class Config:
        from_attributes = True


class PreguntaConEmbedding(MensajePreguntaResponse):
    """Para búsquedas de similitud"""
    similitud: Optional[float] = None