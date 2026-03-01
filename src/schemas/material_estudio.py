from pydantic import BaseModel
from typing import Optional, List

class MaterialBase(BaseModel):
    documento_id: int

class MaterialCreate(BaseModel):
    id: int
    documento_id: int
    material_embedding: list[float]
class MaterialResponse(MaterialBase):
    id: int
    class Config:
        from_attributes = True