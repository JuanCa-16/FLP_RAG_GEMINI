from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List

from src.database.database import SessionLocal
from src.models.biblioteca import Biblioteca
from src.models.documento import Documento
from src.models.material_estudio import MaterialEstudio
from src.schemas.biblioteca import BibliotecaCreate, BibliotecaResponse

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 🔹 Agregar documento (y opcionalmente fragmento) a la biblioteca
@router.post("/", response_model=BibliotecaResponse)
def agregar_a_biblioteca(biblioteca: BibliotecaCreate, db: Session = Depends(get_db)):

    # Verificar documento
    documento = db.query(Documento).filter(
        Documento.id == biblioteca.documento_id
    ).first()

    if not documento:
        raise HTTPException(status_code=404, detail="Documento no encontrado")

    # Si viene material_id verificar que pertenezca al documento
    if biblioteca.material_id:
        material = db.query(MaterialEstudio).filter(
            MaterialEstudio.id == biblioteca.material_id,
            MaterialEstudio.documento_id == biblioteca.documento_id
        ).first()

        if not material:
            raise HTTPException(status_code=404, detail="Material no válido para este documento")

    # Evitar duplicado exacto
    existente = db.query(Biblioteca).filter(
        Biblioteca.usuario == biblioteca.usuario,
        Biblioteca.documento_id == biblioteca.documento_id,
        Biblioteca.material_id == biblioteca.material_id
    ).first()

    if existente:
        raise HTTPException(
            status_code=400,
            detail="Ya está guardado en la biblioteca"
        )

    nueva_entrada = Biblioteca(
        usuario=biblioteca.usuario,
        origen=biblioteca.origen,
        documento_id=biblioteca.documento_id,
        material_id=biblioteca.material_id
    )

    db.add(nueva_entrada)
    db.commit()
    db.refresh(nueva_entrada)

    return nueva_entrada


# 🔹 Obtener biblioteca del usuario
@router.get("/usuario/{username}")
def obtener_biblioteca_usuario(username: str, db: Session = Depends(get_db)):

    resultados = (
        db.query(Biblioteca, Documento, MaterialEstudio)
        .join(Documento, Biblioteca.documento_id == Documento.id)
        .outerjoin(MaterialEstudio, Biblioteca.material_id == MaterialEstudio.id)
        .filter(Biblioteca.usuario == username)
        .order_by(Biblioteca.fecha_consulta.desc())
        .all()
    )

    response = []

    for entrada, documento, material in resultados:
        response.append({
            "id": entrada.id,
            "origen": entrada.origen,
            "fecha_consulta": entrada.fecha_consulta,
            "documento": {
                "id": documento.id,
                "fuente": documento.fuente,
                "nombre_documento": documento.nombre_documento,
                "url": documento.url,
                "tematica": documento.tematica,
                "nivel_dificultad": documento.nivel_dificultad
            },
            "material_id": material.id if material else None
        })

    return response


# 🔹 Verificar si un documento (o fragmento) está en biblioteca
@router.get("/usuario/{username}/documento/{documento_id}")
def verificar_en_biblioteca(username: str, documento_id: int, db: Session = Depends(get_db)):

    existe = db.query(Biblioteca).filter(
        Biblioteca.usuario == username,
        Biblioteca.documento_id == documento_id
    ).first()

    return {"en_biblioteca": existe is not None}


# 🔹 Eliminar por id
@router.delete("/{biblioteca_id}")
def eliminar_de_biblioteca(biblioteca_id: int, db: Session = Depends(get_db)):

    entrada = db.query(Biblioteca).filter(
        Biblioteca.id == biblioteca_id
    ).first()

    if not entrada:
        raise HTTPException(status_code=404, detail="Entrada no encontrada")

    db.delete(entrada)
    db.commit()

    return {"message": "Eliminado correctamente"}


# 🔹 Estadísticas
@router.get("/usuario/{username}/estadisticas")
def estadisticas_biblioteca(username: str, db: Session = Depends(get_db)):

    total = db.query(func.count(Biblioteca.id)).filter(
        Biblioteca.usuario == username
    ).scalar()

    por_fuente = (
        db.query(
            Documento.fuente,
            func.count(Biblioteca.id)
        )
        .join(Documento, Biblioteca.documento_id == Documento.id)
        .filter(Biblioteca.usuario == username)
        .group_by(Documento.fuente)
        .all()
    )

    por_nivel = (
        db.query(
            Documento.nivel_dificultad,
            func.count(Biblioteca.id)
        )
        .join(Documento, Biblioteca.documento_id == Documento.id)
        .filter(Biblioteca.usuario == username)
        .group_by(Documento.nivel_dificultad)
        .all()
    )

    return {
        "total": total,
        "por_fuente": [{"fuente": f, "cantidad": c} for f, c in por_fuente],
        "por_nivel": [{"nivel": n, "cantidad": c} for n, c in por_nivel]
    }