# src/schemas/respuesta_material.py

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class RespuestaMaterialBase(BaseModel):
    """Schema base para relación respuesta-material"""
    material_id: int
    similitud: Optional[float] = Field(None, ge=0.0, le=1.0, description="Similitud entre 0 y 1")
    orden: Optional[int] = Field(None, ge=1, description="Orden de recomendación (1 = más relevante)")


class RespuestaMaterialCreate(RespuestaMaterialBase):
    """Schema para crear asociación respuesta-material"""
    mensaje_respuesta_id: int


class RespuestaMaterialUpdate(BaseModel):
    """Schema para actualizar similitud u orden"""
    similitud: Optional[float] = Field(None, ge=0.0, le=1.0)
    orden: Optional[int] = Field(None, ge=1)


class RespuestaMaterialResponse(RespuestaMaterialBase):
    """Schema para respuesta con datos completos"""
    id: int
    mensaje_respuesta_id: int
    fecha_asociacion: datetime

    class Config:
        from_attributes = True


class MaterialConSimilitud(BaseModel):
    """Schema para material con su similitud"""
    id: int
    fuente: str
    nombre_documento: str
    url: Optional[str]
    tematica: Optional[str]
    nivel_dificultad: Optional[str]
    similitud: float
    orden: Optional[int]

    class Config:
        from_attributes = True