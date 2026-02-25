from pydantic import BaseModel
from typing import Optional, Literal
from datetime import datetime

class DocumentoBase(BaseModel):
    fuente: Literal['CODIGO', 'PDF', 'VIDEO', 'GIT']
    nombre_documento: Optional[str] = None
    url: Optional[str] = None
    tematica: Optional[str] = None
    competencia: Optional[str] = None
    resultado_aprendizaje: Optional[str] = None
    nivel_dificultad: Optional[Literal['Baja', 'Media', 'Alta']] = None

class DocumentoCreate(DocumentoBase):
    pass

class DocumentoResponse(DocumentoBase):
    id: int
    fecha_creacion: datetime

    class Config:
        from_attributes = True