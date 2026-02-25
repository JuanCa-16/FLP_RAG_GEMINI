from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from src.database.database import SessionLocal
from src.models.material_estudio import MaterialEstudio
from src.models.documento import Documento
from src.schemas.material_estudio import MaterialCreate, MaterialResponse

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 🔹 Crear fragmento (embedding) asociado a un documento
@router.post("/", response_model=MaterialResponse)
def crear_material(material: MaterialCreate, db: Session = Depends(get_db)):

    # Verificar que el documento exista
    documento = db.query(Documento).filter(
        Documento.id == material.documento_id
    ).first()

    if not documento:
        raise HTTPException(status_code=404, detail="Documento no encontrado")

    # 🔎 Verificar que el ID no esté repetido
    existe = db.query(MaterialEstudio).filter(
        MaterialEstudio.id == material.id
    ).first()

    if existe:
        raise HTTPException(status_code=400, detail="Ya existe un material con ese ID")

    nuevo_material = MaterialEstudio(
        id=material.id,  # ⭐ ahora lo envías explícitamente
        documento_id=material.documento_id,
        material_embedding=material.material_embedding
    )

    db.add(nuevo_material)
    db.commit()
    db.refresh(nuevo_material)

    return nuevo_material

# 🔹 Listar todos los fragmentos
@router.get("/", response_model=List[MaterialResponse])
def listar_materiales(db: Session = Depends(get_db)):
    return db.query(MaterialEstudio).all()


# 🔹 Obtener fragmento por ID
@router.get("/{material_id}", response_model=MaterialResponse)
def obtener_material(material_id: int, db: Session = Depends(get_db)):

    material = db.query(MaterialEstudio).filter(
        MaterialEstudio.id == material_id
    ).first()

    if not material:
        raise HTTPException(status_code=404, detail="Material no encontrado")

    return material


# 🔹 Obtener todos los fragmentos de un documento
@router.get("/documento/{documento_id}", response_model=List[MaterialResponse])
def obtener_materiales_por_documento(documento_id: int, db: Session = Depends(get_db)):

    materiales = db.query(MaterialEstudio).filter(
        MaterialEstudio.documento_id == documento_id
    ).all()

    return materiales


# 🔹 Eliminar fragmento
@router.delete("/{material_id}")
def eliminar_material(material_id: int, db: Session = Depends(get_db)):

    material = db.query(MaterialEstudio).filter(
        MaterialEstudio.id == material_id
    ).first()

    if not material:
        raise HTTPException(status_code=404, detail="Material no encontrado")

    db.delete(material)
    db.commit()

    return {"message": "Material eliminado correctamente"}