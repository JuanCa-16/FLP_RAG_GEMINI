# src/schemas/material.py

from pydantic import BaseModel
from typing import Optional, List, Literal
from datetime import datetime



class MaterialBase(BaseModel):
    fuente: Literal['CODIGO', 'PDF', 'VIDEO', 'GIT']
    nombre_documento: Optional[str] = None
    url: Optional[str] = None
    tematica: Optional[str] = None
    competencia: Optional[str] = None
    resultado_aprendizaje: Optional[str] = None
    nivel_dificultad: Optional[Literal['Baja', 'Media', 'Alta']] = None


class MaterialCreate(MaterialBase):
    # embedding opcional al crear
    material_embedding: Optional[List[float]] = None


class MaterialResponse(MaterialBase):
    id: int
    fecha_creacion: datetime

    class Config:
        from_attributes = True