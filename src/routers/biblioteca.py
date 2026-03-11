from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List

from src.database.database import SessionLocal
from src.models.biblioteca import Biblioteca
from src.models.documento import Documento
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


# 🔹 Agregar documento (y opcionalmente fragmento) a la biblioteca
@router.post("/", response_model=BibliotecaResponse)
def agregar_a_biblioteca(biblioteca: BibliotecaCreate,current_user: str = Depends(get_current_user), db: Session = Depends(get_db)):

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

    nueva_entrada = Biblioteca(
        usuario=current_user.usuario,
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

# ============================================
# HISTORIAL IMPLÍCITO PARA RECOMENDADOR
# ============================================

@router.get("/usuario/{username}/historial-implicito")
def obtener_historial_implicito(username: str, limit: int = 1, db: Session = Depends(get_db)):
    """
    Obtiene el historial implícito del usuario basado en su biblioteca.
    Distingue entre fragmentos específicos y documentos completos.
    """
    
    # Obtener TODAS las entradas (con y sin material_id)
    resultados = (
        db.query(
            Biblioteca.material_id,
            Biblioteca.documento_id,
            Biblioteca.fecha_consulta,
            MaterialEstudio.material_embedding
        )
        .outerjoin(MaterialEstudio, MaterialEstudio.id == Biblioteca.material_id)
        .filter(Biblioteca.usuario == username)
        .order_by(Biblioteca.fecha_consulta.desc())
        .limit(limit)
        .all()
    )
    
    return [
        {
            "tipo": "fragmento" if material_id else "documento_completo",
            "material_id": material_id,
            "documento_id": documento_id,
            "fecha_consulta": fecha_consulta,
            "material_embedding": [float(x) for x in material_embedding] if material_embedding is not None else None
        }
        for material_id, documento_id, fecha_consulta, material_embedding in resultados
    ]