from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Literal

class BibliotecaBase(BaseModel):
    usuario: str
    origen: Literal['CHAT', 'PDF', 'VIDEO']
    documento_id: int
    material_id: Optional[int] = None

class BibliotecaCreate(BibliotecaBase):
    pass

class BibliotecaResponse(BibliotecaBase):
    id: int
    fecha_consulta: datetime

    class Config:
        from_attributes = True