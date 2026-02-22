# src/schemas/biblioteca.py

from pydantic import BaseModel
from datetime import datetime


class BibliotecaBase(BaseModel):
    usuario: str
    material_id: int


class BibliotecaCreate(BibliotecaBase):
    pass


class BibliotecaResponse(BibliotecaBase):
    id: int
    fecha_consulta: datetime

    class Config:
        from_attributes = True