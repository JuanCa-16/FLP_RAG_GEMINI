from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional

from src.database.database import SessionLocal
from src.models.documento import Documento
from src.schemas.documento import DocumentoCreate, DocumentoResponse

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 🔹 Crear documento
@router.post("/", response_model=DocumentoResponse)
def crear_documento(documento: DocumentoCreate, db: Session = Depends(get_db)):

    nuevo_documento = Documento(
        fuente=documento.fuente,
        nombre_documento=documento.nombre_documento,
        url=documento.url,
        tematica=documento.tematica,
        competencia=documento.competencia,
        resultado_aprendizaje=documento.resultado_apendizaje,
        nivel_dificultad=documento.nivel_dificultad
    )

    db.add(nuevo_documento)
    db.commit()
    db.refresh(nuevo_documento)

    return nuevo_documento


# 🔹 Listar todos
@router.get("/", response_model=List[DocumentoResponse])
def listar_documentos(
    fuente: Optional[str] = None,
    nivel: Optional[str] = None,
    db: Session = Depends(get_db)
):

    query = db.query(Documento)

    if fuente:
        query = query.filter(Documento.fuente == fuente)

    if nivel:
        query = query.filter(Documento.nivel_dificultad == nivel)

    return query.all()


# 🔹 Obtener por ID
@router.get("/{documento_id}", response_model=DocumentoResponse)
def obtener_documento(documento_id: int, db: Session = Depends(get_db)):

    documento = db.query(Documento).filter(
        Documento.id == documento_id
    ).first()

    if not documento:
        raise HTTPException(status_code=404, detail="Documento no encontrado")

    return documento


# 🔹 Actualizar documento
@router.put("/{documento_id}", response_model=DocumentoResponse)
def actualizar_documento(
    documento_id: int,
    datos: DocumentoCreate,
    db: Session = Depends(get_db)
):

    documento = db.query(Documento).filter(
        Documento.id == documento_id
    ).first()

    if not documento:
        raise HTTPException(status_code=404, detail="Documento no encontrado")

    documento.fuente = datos.fuente
    documento.nombre_documento = datos.nombre_documento
    documento.url = datos.url
    documento.tematica = datos.tematica
    documento.competencia = datos.competencia
    documento.resultado_aprendizaje = datos.resultado_apendizaje
    documento.nivel_dificultad = datos.nivel_dificultad

    db.commit()
    db.refresh(documento)

    return documento


# 🔹 Eliminar documento
@router.delete("/{documento_id}")
def eliminar_documento(documento_id: int, db: Session = Depends(get_db)):

    documento = db.query(Documento).filter(
        Documento.id == documento_id
    ).first()

    if not documento:
        raise HTTPException(status_code=404, detail="Documento no encontrado")

    db.delete(documento)
    db.commit()

    return {"message": "Documento eliminado correctamente"}