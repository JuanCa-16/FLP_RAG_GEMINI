from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.database.database import SessionLocal
from src.models.usuario import Usuario
from src.schemas.usuario import UsuarioCreate
from src.core.security import hash_password

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/")
def crear_usuario(data: UsuarioCreate, db: Session = Depends(get_db)):

    existente = db.query(Usuario).filter(
        Usuario.usuario == data.usuario
    ).first()

    if existente:
        raise HTTPException(status_code=400, detail="Usuario ya existe")

    password_hashed = hash_password(data.contrasena)

    nuevo = Usuario(
        usuario=data.usuario,
        contrasena=password_hashed,
        nombre=data.nombre
    )

    db.add(nuevo)
    db.commit()

    return {"mensaje": "Usuario creado correctamente"}
@router.get("/{username}")
def obtener_usuario(username: str, db: Session = Depends(get_db)):
    user = db.query(Usuario).filter(
        Usuario.usuario == username
    ).first()

    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    return {"usuario": user.usuario, "nombre": user.nombre}

@router.put("/{username}")
def actualizar_nombre_usuario(username: str, nombre: str, db: Session = Depends(get_db)):
    user = db.query(Usuario).filter(
        Usuario.usuario == username
    ).first()

    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    user.nombre = nombre
    db.commit()

    return {"mensaje": "Nombre de usuario actualizado"}

@router.delete("/{username}")
def eliminar_usuario(username: str, db: Session = Depends(get_db)):
    user = db.query(Usuario).filter(
        Usuario.usuario == username
    ).first()

    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    db.delete(user)
    db.commit()

    return {"mensaje": "Usuario eliminado con todo su historial"}