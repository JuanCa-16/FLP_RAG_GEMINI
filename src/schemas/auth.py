# src/schemas/auth.py

from pydantic import BaseModel, Field, validator
from typing import Optional


class UsuarioRegistro(BaseModel):
    """Schema para registro de nuevo usuario"""
    usuario: str = Field(..., min_length=3, max_length=15, description="Nombre de usuario único")
    nombre: str = Field(..., min_length=2, max_length=20, description="Nombre completo del usuario")
    contrasena: str = Field(..., min_length=6, max_length=100, description="Contraseña (mínimo 6 caracteres)")
    
    @validator('usuario')
    def validar_usuario(cls, v):
        """Valida que el usuario no tenga espacios"""
        if ' ' in v:
            raise ValueError('El usuario no puede contener espacios')
        return v  # Convertir a minúsculas
    
    @validator('contrasena')
    def validar_contrasena(cls, v):
        """Valida que la contraseña tenga al menos 6 caracteres"""
        if len(v) < 6:
            raise ValueError('La contraseña debe tener al menos 6 caracteres')
        return v


class UsuarioLogin(BaseModel):
    """Schema para inicio de sesión"""
    usuario: str = Field(..., description="Nombre de usuario")
    contrasena: str = Field(..., description="Contraseña")


class Token(BaseModel):
    """Schema para respuesta de token"""
    access_token: str
    token_type: str = "bearer"
    usuario: str
    nombre: str


class TokenData(BaseModel):
    """Schema para datos decodificados del token"""
    usuario: Optional[str] = None


class UsuarioResponse(BaseModel):
    """Schema para respuesta de usuario autenticado"""
    usuario: str
    nombre: str
    
    class Config:
        from_attributes = True


class MensajeResponse(BaseModel):
    """Schema para mensajes de respuesta"""
    mensaje: str
    detalle: Optional[str] = None