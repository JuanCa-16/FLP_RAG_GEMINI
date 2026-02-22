# src/routes/auth.py

from fastapi import APIRouter, Depends, HTTPException, status, Header
from sqlalchemy.orm import Session
from datetime import timedelta
from typing import Optional

from src.database.database import SessionLocal
from src.models.usuario import Usuario
from src.schemas.auth import (
    UsuarioRegistro, 
    UsuarioLogin, 
    Token, 
    UsuarioResponse,
    MensajeResponse
)
from src.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

router = APIRouter()


def get_db():
    """Dependencia para obtener sesión de base de datos"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
) -> Usuario:
    """
    Dependencia para obtener el usuario actual desde el token JWT.
    
    Args:
        authorization: Header Authorization con formato "Bearer <token>"
        db: Sesión de base de datos
        
    Returns:
        Usuario autenticado
        
    Raises:
        HTTPException: Si el token es inválido o el usuario no existe
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No se pudo validar las credenciales",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if not authorization:
        raise credentials_exception
    
    # Extraer el token del header "Bearer <token>"
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise credentials_exception
    except ValueError:
        raise credentials_exception
    
    # Decodificar el token
    usuario_nombre = decode_access_token(token)
    
    if usuario_nombre is None:
        raise credentials_exception
    
    # Buscar el usuario en la base de datos
    usuario = db.query(Usuario).filter(Usuario.usuario == usuario_nombre).first()
    
    if usuario is None:
        raise credentials_exception
    
    return usuario


# ============================================
# 1. REGISTRO (CREAR CUENTA)
# ============================================
@router.post("/registro", response_model=Token, status_code=status.HTTP_201_CREATED)
def registrar_usuario(
    datos: UsuarioRegistro,
    db: Session = Depends(get_db)
):
    """
    Registra un nuevo usuario en el sistema.
    
    - **usuario**: Nombre de usuario único (3-15 caracteres, sin espacios)
    - **nombre**: Nombre completo (2-20 caracteres)
    - **contrasena**: Contraseña (mínimo 6 caracteres)
    
    Retorna un token JWT para inicio de sesión automático.
    """
    # Verificar si el usuario ya existe
    usuario_existente = db.query(Usuario).filter(
        Usuario.usuario == datos.usuario.lower()
    ).first()
    
    if usuario_existente:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"El usuario '{datos.usuario}' ya existe"
        )
    
    # Hashear la contraseña
    password_hash = hash_password(datos.contrasena)
    
    # Crear nuevo usuario
    nuevo_usuario = Usuario(
        usuario=datos.usuario.lower(),
        nombre=datos.nombre,
        contrasena=password_hash
    )
    
    db.add(nuevo_usuario)
    db.commit()
    db.refresh(nuevo_usuario)
    
    # Crear token de acceso
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": nuevo_usuario.usuario},
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "usuario": nuevo_usuario.usuario,
        "nombre": nuevo_usuario.nombre
    }


# ============================================
# 2. INICIAR SESIÓN (LOGIN)
# ============================================
@router.post("/login", response_model=Token)
def iniciar_sesion(
    credenciales: UsuarioLogin,
    db: Session = Depends(get_db)
):
    """
    Inicia sesión con usuario y contraseña.
    
    - **usuario**: Nombre de usuario
    - **contrasena**: Contraseña
    
    Retorna un token JWT válido por 24 horas.
    """
    # Buscar el usuario
    usuario = db.query(Usuario).filter(
        Usuario.usuario == credenciales.usuario.lower()
    ).first()
    
    if not usuario:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario o contraseña incorrectos",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verificar la contraseña
    if not verify_password(credenciales.contrasena, usuario.contrasena):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario o contraseña incorrectos",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Crear token de acceso
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": usuario.usuario},
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "usuario": usuario.usuario,
        "nombre": usuario.nombre
    }


# ============================================
# 3. CERRAR SESIÓN (LOGOUT)
# ============================================
@router.post("/logout", response_model=MensajeResponse)
def cerrar_sesion(
    current_user: Usuario = Depends(get_current_user)
):
    """
    Cierra la sesión del usuario actual.
    
    Nota: En JWT, el logout se maneja principalmente del lado del cliente
    eliminando el token. Este endpoint sirve para confirmar el cierre de sesión
    y podría usarse para invalidar tokens en una lista negra (blacklist) si se implementa.
    
    Requiere token JWT válido en el header Authorization.
    """
    return {
        "mensaje": "Sesión cerrada correctamente",
        "detalle": f"Usuario '{current_user.usuario}' ha cerrado sesión"
    }


# ============================================
# 4. OBTENER USUARIO ACTUAL (ME)
# ============================================
@router.get("/me", response_model=UsuarioResponse)
def obtener_usuario_actual(
    current_user: Usuario = Depends(get_current_user)
):
    """
    Obtiene la información del usuario actualmente autenticado.
    
    Requiere token JWT válido en el header Authorization.
    """
    return {
        "usuario": current_user.usuario,
        "nombre": current_user.nombre
    }


# ============================================
# 5. VERIFICAR TOKEN
# ============================================
@router.post("/verificar-token")
def verificar_token(
    current_user: Usuario = Depends(get_current_user)
):
    """
    Verifica si un token JWT es válido.
    
    Útil para validar tokens antes de hacer requests importantes.
    Requiere token JWT en el header Authorization.
    """
    return {
        "valido": True,
        "usuario": current_user.usuario,
        "nombre": current_user.nombre,
        "mensaje": "Token válido"
    }


# ============================================
# 6. CAMBIAR CONTRASEÑA
# ============================================
@router.put("/cambiar-contrasena", response_model=MensajeResponse)
def cambiar_contrasena(
    contrasena_actual: str,
    contrasena_nueva: str,
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Cambia la contraseña del usuario actual.
    
    - **contrasena_actual**: Contraseña actual del usuario
    - **contrasena_nueva**: Nueva contraseña (mínimo 6 caracteres)
    
    Requiere token JWT válido.
    """
    # Verificar contraseña actual
    if not verify_password(contrasena_actual, current_user.contrasena):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="La contraseña actual es incorrecta"
        )
    
    # Validar nueva contraseña
    if len(contrasena_nueva) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La nueva contraseña debe tener al menos 6 caracteres"
        )
    
    # Actualizar contraseña
    current_user.contrasena = hash_password(contrasena_nueva)
    db.commit()
    
    return {
        "mensaje": "Contraseña actualizada correctamente",
        "detalle": "Deberás iniciar sesión nuevamente con tu nueva contraseña"
    }