# src/routes/material.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from src.database.database import SessionLocal
from src.models.material_estudio import MaterialEstudio
from src.schemas.material_estudio import MaterialCreate, MaterialResponse

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 🔹 Crear material
@router.post("/", response_model=MaterialResponse)
def crear_material(material: MaterialCreate, db: Session = Depends(get_db)):
    
    nuevo_material = MaterialEstudio(
        fuente=material.fuente.value,
        nombre_documento=material.nombre_documento,
        url=material.url,
        tematica=material.tematica,
        competencia=material.competencia,
        resultado_aprendizaje=material.resultado_aprendizaje,
        nivel_dificultad=material.nivel_dificultad.value,
        material_embedding=material.material_embedding
    )

    db.add(nuevo_material)
    db.commit()
    db.refresh(nuevo_material)

    return nuevo_material


# 🔹 Listar todos
@router.get("/", response_model=List[MaterialResponse])
def listar_materiales(db: Session = Depends(get_db)):
    return db.query(MaterialEstudio).all()


# 🔹 Obtener por ID
@router.get("/{material_id}", response_model=MaterialResponse)
def obtener_material(material_id: int, db: Session = Depends(get_db)):

    material = (
        db.query(MaterialEstudio)
        .filter(MaterialEstudio.id == material_id)
        .first()
    )

    if not material:
        raise HTTPException(status_code=404, detail="Material no encontrado")

    return material


# 🔹 Eliminar
@router.delete("/{material_id}")
def eliminar_material(material_id: int, db: Session = Depends(get_db)):

    material = (
        db.query(MaterialEstudio)
        .filter(MaterialEstudio.id == material_id)
        .first()
    )

    if not material:
        raise HTTPException(status_code=404, detail="Material no encontrado")

    db.delete(material)
    db.commit()

    return {"message": "Material eliminado correctamente"}