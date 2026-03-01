# src/routes/auth.py (ACTUALIZADO PARA SWAGGER UI)

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials  # ⭐ IMPORTANTE
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

# ⭐ ESQUEMA DE SEGURIDAD PARA SWAGGER UI
security = HTTPBearer(
    scheme_name="Bearer Authentication",
    description="Ingresa tu token JWT (obtenido del login)"
)


def get_db():
    """Dependencia para obtener sesión de base de datos"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),  # ⭐ CAMBIO AQUÍ
    db: Session = Depends(get_db)
) -> Usuario:
    """
    Dependencia para obtener el usuario actual desde el token JWT.
    
    **Uso en Swagger UI:**
    1. Haz login y copia el access_token
    2. Click en "Authorize" 🔓
    3. Pega el token
    4. Click "Authorize"
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No se pudo validar las credenciales",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Extraer el token de las credenciales
    token = credentials.credentials
    
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
@router.post(
    "/registro",
    response_model=Token,
    status_code=status.HTTP_201_CREATED,
    summary="Registrar nuevo usuario",
    description="""
    Crea una nueva cuenta de usuario y retorna un token JWT.
    
    **Ejemplo:**
    ```json
    {
      "usuario": "juan123",
      "nombre": "Juan Pérez",
      "contrasena": "password123"
    }
    ```
    
    **Respuesta:**
    - access_token: Token JWT para autenticación
    - token_type: Siempre "bearer"
    - usuario: Nombre de usuario
    - nombre: Nombre completo
    """
)
def registrar_usuario(
    datos: UsuarioRegistro,
    db: Session = Depends(get_db)
):
    # Verificar si el usuario ya existe
    usuario_existente = db.query(Usuario).filter(
        Usuario.usuario == datos.usuario
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
        usuario=datos.usuario,
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
@router.post(
    "/login",
    response_model=Token,
    summary="Iniciar sesión",
    description="""
    Inicia sesión y obtén un token JWT válido por 24 horas.
    
    **Pasos para usar en Swagger:**
    1. Ejecuta este endpoint con tu usuario y contraseña
    2. Copia el `access_token` de la respuesta
    3. Click en el botón "Authorize" 🔓 (arriba a la derecha)
    4. Pega el token (sin "Bearer")
    5. Click "Authorize"
    6. ¡Ahora puedes usar todos los endpoints protegidos!
    
    **Ejemplo:**
    ```json
    {
      "usuario": "juan123",
      "contrasena": "password123"
    }
    ```
    """
)
def iniciar_sesion(
    credenciales: UsuarioLogin,
    db: Session = Depends(get_db)
):
    # Buscar el usuario
    usuario = db.query(Usuario).filter(
        Usuario.usuario == credenciales.usuario
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
@router.post(
    "/logout",
    response_model=MensajeResponse,
    summary="Cerrar sesión",
    description="Cierra la sesión del usuario actual. Requiere token JWT."
)
def cerrar_sesion(
    current_user: Usuario = Depends(get_current_user)
):
    """
    Cierra la sesión del usuario actual.
    
    Nota: En JWT, el logout se maneja del lado del cliente eliminando el token.
    """
    return {
        "mensaje": "Sesión cerrada correctamente",
        "detalle": f"Usuario '{current_user.usuario}' ha cerrado sesión"
    }


# ============================================
# 4. OBTENER USUARIO ACTUAL (ME)
# ============================================
@router.get(
    "/me",
    response_model=UsuarioResponse,
    summary="Obtener usuario actual",
    description="Retorna la información del usuario autenticado. Requiere token JWT."
)
def obtener_usuario_actual(
    current_user: Usuario = Depends(get_current_user)
):
    """
    Obtiene la información del usuario actualmente autenticado.
    """
    return {
        "usuario": current_user.usuario,
        "nombre": current_user.nombre
    }


# ============================================
# 5. VERIFICAR TOKEN
# ============================================
@router.post(
    "/verificar-token",
    summary="Verificar validez del token",
    description="Verifica si un token JWT es válido. Requiere token JWT."
)
def verificar_token(
    current_user: Usuario = Depends(get_current_user)
):
    """
    Verifica si un token JWT es válido.
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
@router.put(
    "/cambiar-contrasena",
    response_model=MensajeResponse,
    summary="Cambiar contraseña",
    description="Cambia la contraseña del usuario actual. Requiere token JWT."
)
def cambiar_contrasena(
    contrasena_actual: str,
    contrasena_nueva: str,
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Cambia la contraseña del usuario actual.
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