from pydantic import BaseModel

class UsuarioCreate(BaseModel):
    usuario: str
    contrasena: str
    nombre: str