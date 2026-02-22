# src/routes/biblioteca.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from src.database.database import SessionLocal
from src.models.biblioteca import Biblioteca
from src.models.material_estudio import MaterialEstudio
from src.schemas.biblioteca import BibliotecaCreate, BibliotecaResponse
from src.routers.auth import get_current_user
router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 🔹 Agregar material a la biblioteca del usuario
@router.post("/", response_model=BibliotecaResponse)
def agregar_a_biblioteca(biblioteca: BibliotecaCreate, db: Session = Depends(get_db)):
    # Verificar que el material existe
    material = db.query(MaterialEstudio).filter(
        MaterialEstudio.id == biblioteca.material_id
    ).first()
    
    if not material:
        raise HTTPException(status_code=404, detail="Material no encontrado")
    
    # Verificar si ya existe en la biblioteca del usuario
    existente = db.query(Biblioteca).filter(
        Biblioteca.usuario == biblioteca.usuario,
        Biblioteca.material_id == biblioteca.material_id
    ).first()
    
    if existente:
        raise HTTPException(
            status_code=400,
            detail="Este material ya está en tu biblioteca"
        )
    
    # Crear entrada en biblioteca
    nueva_entrada = Biblioteca(
        usuario=biblioteca.usuario,
        material_id=biblioteca.material_id
    )
    
    db.add(nueva_entrada)
    db.commit()
    db.refresh(nueva_entrada)
    
    return nueva_entrada


# 🔹 Obtener biblioteca de un usuario
@router.get("/usuario/{username}")
def obtener_biblioteca_usuario(username: str, db: Session = Depends(get_db)):
    biblioteca = (
        db.query(Biblioteca, MaterialEstudio)
        .join(MaterialEstudio, Biblioteca.material_id == MaterialEstudio.id)
        .filter(Biblioteca.usuario == username)
        .order_by(Biblioteca.fecha_consulta.desc())
        .all()
    )
    
    resultado = []
    for entrada, material in biblioteca:
        resultado.append({
            "id": entrada.id,
            "fecha_consulta": entrada.fecha_consulta,
            "material": {
                "id": material.id,
                "fuente": material.fuente,
                "nombre_documento": material.nombre_documento,
                "url": material.url,
                "tematica": material.tematica,
                "nivel_dificultad": material.nivel_dificultad
            }
        })
    
    return resultado


# 🔹 Verificar si un material está en la biblioteca
@router.get("/usuario/{username}/tiene/{material_id}")
def verificar_en_biblioteca(username: str, material_id: int, db: Session = Depends(get_db)):
    existe = db.query(Biblioteca).filter(
        Biblioteca.usuario == username,
        Biblioteca.material_id == material_id
    ).first()
    
    return {"en_biblioteca": existe is not None}


# 🔹 Eliminar material de la biblioteca
@router.delete("/{biblioteca_id}")
def eliminar_de_biblioteca(biblioteca_id: int, db: Session = Depends(get_db)):
    entrada = db.query(Biblioteca).filter(Biblioteca.id == biblioteca_id).first()
    
    if not entrada:
        raise HTTPException(status_code=404, detail="Entrada no encontrada")
    
    db.delete(entrada)
    db.commit()
    
    return {"message": "Material eliminado de la biblioteca"}


# 🔹 Eliminar por usuario y material_id
@router.delete("/usuario/{username}/material/{material_id}")
def eliminar_material_usuario(username: str, material_id: int, db: Session = Depends(get_db)):
    entrada = db.query(Biblioteca).filter(
        Biblioteca.usuario == username,
        Biblioteca.material_id == material_id
    ).first()
    
    if not entrada:
        raise HTTPException(
            status_code=404,
            detail="Este material no está en tu biblioteca"
        )
    
    db.delete(entrada)
    db.commit()
    
    return {"message": "Material eliminado de la biblioteca"}


# 🔹 Obtener estadísticas de la biblioteca
@router.get("/usuario/{username}/estadisticas")
def estadisticas_biblioteca(username: str, db: Session = Depends(get_db)):
    from sqlalchemy import func
    
    # Total de materiales
    total = db.query(func.count(Biblioteca.id)).filter(
        Biblioteca.usuario == username
    ).scalar()
    
    # Por fuente
    por_fuente = (
        db.query(
            MaterialEstudio.fuente,
            func.count(Biblioteca.id).label("cantidad")
        )
        .join(MaterialEstudio, Biblioteca.material_id == MaterialEstudio.id)
        .filter(Biblioteca.usuario == username)
        .group_by(MaterialEstudio.fuente)
        .all()
    )
    
    # Por nivel
    por_nivel = (
        db.query(
            MaterialEstudio.nivel_dificultad,
            func.count(Biblioteca.id).label("cantidad")
        )
        .join(MaterialEstudio, Biblioteca.material_id == MaterialEstudio.id)
        .filter(Biblioteca.usuario == username)
        .group_by(MaterialEstudio.nivel_dificultad)
        .all()
    )
    
    return {
        "total_materiales": total,
        "por_fuente": [{"fuente": f, "cantidad": c} for f, c in por_fuente],
        "por_nivel": [{"nivel": n, "cantidad": c} for n, c in por_nivel]
    }