# src/routes/mensaje.py (ACTUALIZADO CON MÚLTIPLES MATERIALES)

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional

from src.database.database import SessionLocal
from src.models.mensaje import Mensaje
from src.models.mensaje_pregunta import MensajePregunta
from src.models.mensaje_respuesta import MensajeRespuesta
from src.models.respuesta_material import RespuestaMaterial
from src.models.material_estudio import MaterialEstudio
from src.models.chat import Chat
from src.models.documento import Documento

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================
# CREAR MENSAJE CON RESPUESTA Y MÚLTIPLES MATERIALES
# ============================================
@router.post("/respuesta-con-materiales")
def crear_respuesta_con_materiales(
    chat_id: int,
    rol: str,
    tipo: str,
    respuesta: str,
    materiales: List[dict],  # [{"material_id": 1, "similitud": 0.95, "orden": 1}, ...]
    db: Session = Depends(get_db)
):
    """
    Crea un mensaje de respuesta con múltiples materiales de estudio asociados.
    
    Args:
        chat_id: ID del chat
        rol: 'assistant' o 'user'
        tipo: 'CODIGO', 'PDF', 'VIDEO', 'GIT'
        respuesta: Texto de la respuesta
        materiales: Lista de diccionarios con:
            - material_id: ID del material
            - similitud: Float entre 0 y 1 (opcional)
            - orden: Int para ordenar (opcional, 1 = más relevante)
    
    Ejemplo de materiales:
    [
        {"material_id": 5, "similitud": 0.95, "orden": 1},
        {"material_id": 12, "similitud": 0.87, "orden": 2},
        {"material_id": 3, "similitud": 0.82, "orden": 3}
    ]
    """
    # Verificar que el chat existe
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat no encontrado")
    
    # 1. Crear mensaje
    nuevo_mensaje = Mensaje(
        chat_id=chat_id,
        rol=rol,
        tipo=tipo
    )
    db.add(nuevo_mensaje)
    db.flush()
    
    # 2. Crear respuesta asociada
    nueva_respuesta = MensajeRespuesta(
        mensaje_id=nuevo_mensaje.id,
        respuesta=respuesta
    )
    db.add(nueva_respuesta)
    db.flush()
    
    # 3. Asociar múltiples materiales con sus similitudes
    materiales_asociados = []
    for idx, mat in enumerate(materiales, 1):
        # Verificar que el material existe
        material = db.query(MaterialEstudio).filter(
            MaterialEstudio.id == mat["material_id"]
        ).first()
        
        if not material:
            raise HTTPException(
                status_code=404,
                detail=f"Material con ID {mat['material_id']} no encontrado"
            )
        
        documento = db.query(Documento).filter(
            Documento.id == material.documento_id
        ).first()

        if not documento:
            raise HTTPException(
                status_code=404,
                detail=f"Documento asociado no encontrado para material {material.id}"
            )
        
        # Crear asociación
        asociacion = RespuestaMaterial(
            mensaje_respuesta_id=nueva_respuesta.id,
            material_id=mat["material_id"],
            similitud=mat.get("similitud"),
            orden=mat.get("orden", idx)  # Si no se proporciona, usar índice
        )

        db.add(asociacion)
        materiales_asociados.append({
            "material_id": mat["material_id"],
            "documento_id": documento.id,
            "nombre": documento.nombre_documento,
            "similitud": mat.get("similitud"),
            "orden": mat.get("orden", idx)
        })
    
    db.commit()
    db.refresh(nuevo_mensaje)
    
    return {
        "mensaje_id": nuevo_mensaje.id,
        "chat_id": chat_id,
        "respuesta": respuesta,
        "total_materiales": len(materiales_asociados),
        "materiales": materiales_asociados,
        "fecha_mensaje": nuevo_mensaje.fecha_mensaje
    }


# ============================================
# AGREGAR MATERIALES A RESPUESTA EXISTENTE
# ============================================
@router.post("/respuesta/{mensaje_id}/agregar-materiales")
def agregar_materiales_a_respuesta(
    mensaje_id: int,
    materiales: List[dict],
    db: Session = Depends(get_db)
):
    """
    Agrega materiales adicionales a una respuesta existente.
    
    Args:
        mensaje_id: ID del mensaje respuesta
        materiales: Lista con material_id, similitud, orden
    """
    # Verificar que la respuesta existe
    respuesta = db.query(MensajeRespuesta).filter(
        MensajeRespuesta.mensaje_id == mensaje_id
    ).first()
    
    if not respuesta:
        raise HTTPException(status_code=404, detail="Respuesta no encontrada")
    
    materiales_agregados = []
    for mat in materiales:
        # Verificar que el material existe
        material = db.query(MaterialEstudio).filter(
            MaterialEstudio.id == mat["material_id"]
        ).first()
        
        if not material:
            raise HTTPException(
                status_code=404,
                detail=f"Material {mat['material_id']} no encontrado"
            )
        
        documento = db.query(Documento).filter(
            Documento.id == material.documento_id
        ).first()

        if not documento:
            raise HTTPException(
                status_code=404,
                detail=f"Documento asociado no encontrado para material {material.id}"
            )
        
        # Verificar que no esté ya asociado
        existe = db.query(RespuestaMaterial).filter(
            RespuestaMaterial.mensaje_respuesta_id == respuesta.id,
            RespuestaMaterial.material_id == mat["material_id"]
        ).first()
        
        if existe:
            continue  # Skip duplicados
        
        # Crear asociación
        asociacion = RespuestaMaterial(
            mensaje_respuesta_id=respuesta.id,
            material_id=material.id,
            similitud=mat.get("similitud"),
            orden=mat.get("orden")
        )
        db.add(asociacion)
        materiales_agregados.append(documento.nombre_documento)
    
    db.commit()
    
    return {
        "mensaje_id": mensaje_id,
        "materiales_agregados": len(materiales_agregados),
        "nombres": materiales_agregados
    }


# ============================================
# OBTENER MATERIALES DE UNA RESPUESTA
# ============================================
@router.get("/respuesta/{mensaje_id}/materiales")
def obtener_materiales_respuesta(
    mensaje_id: int,
    db: Session = Depends(get_db)
):
    """
    Obtiene todos los materiales asociados a una respuesta con sus similitudes.
    """
    # Verificar que la respuesta existe
    respuesta = db.query(MensajeRespuesta).filter(
        MensajeRespuesta.mensaje_id == mensaje_id
    ).first()
    
    if not respuesta:
        raise HTTPException(status_code=404, detail="Respuesta no encontrada")
    
    # Obtener materiales con sus similitudes
    materiales = (
        db.query(RespuestaMaterial, MaterialEstudio,Documento)
        .join(MaterialEstudio, RespuestaMaterial.material_id == MaterialEstudio.id)
        .join(Documento, MaterialEstudio.documento_id == Documento.id)
        .filter(RespuestaMaterial.mensaje_respuesta_id == respuesta.id)
        .order_by(RespuestaMaterial.orden, RespuestaMaterial.similitud.desc())
        .all()
    )
    
    resultado = []
    for asociacion, material,documento  in materiales:
        resultado.append({
            "material_id": material.id,
            "documento_id": documento.id,
            "nombre_documento": documento.nombre_documento,
            "fuente": documento.fuente,
            "url": documento.url,
            "tematica": documento.tematica,
            "nivel_dificultad": documento.nivel_dificultad,
            "similitud": asociacion.similitud,
            "orden": asociacion.orden,
            "fecha_asociacion": asociacion.fecha_asociacion
        })
    
    return {
        "mensaje_id": mensaje_id,
        "respuesta": respuesta.respuesta,
        "total_materiales": len(resultado),
        "materiales": resultado
    }


# ============================================
# ACTUALIZAR SIMILITUD DE UN MATERIAL
# ============================================
@router.put("/respuesta/{mensaje_id}/material/{material_id}")
def actualizar_similitud_material(
    mensaje_id: int,
    material_id: int,
    similitud: Optional[float] = None,
    orden: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """
    Actualiza la similitud u orden de un material específico en una respuesta.
    """
    asociacion = db.query(RespuestaMaterial).filter(
        RespuestaMaterial.mensaje_respuesta_id == mensaje_id,
        RespuestaMaterial.material_id == material_id
    ).first()
    
    if not asociacion:
        raise HTTPException(
            status_code=404,
            detail="Asociación no encontrada"
        )
    
    if similitud is not None:
        if not (0 <= similitud <= 1):
            raise HTTPException(
                status_code=400,
                detail="La similitud debe estar entre 0 y 1"
            )
        asociacion.similitud = similitud
    
    if orden is not None:
        asociacion.orden = orden
    
    db.commit()
    
    return {
        "mensaje_id": mensaje_id,
        "material_id": material_id,
        "similitud": asociacion.similitud,
        "orden": asociacion.orden
    }


# ============================================
# ELIMINAR MATERIAL DE UNA RESPUESTA
# ============================================
@router.delete("/respuesta/{mensaje_id}/material/{material_id}")
def eliminar_material_de_respuesta(
    mensaje_id: int,
    material_id: int,
    db: Session = Depends(get_db)
):
    """
    Elimina un material específico de una respuesta.
    """
    asociacion = db.query(RespuestaMaterial).filter(
        RespuestaMaterial.mensaje_respuesta_id == mensaje_id,
        RespuestaMaterial.material_id == material_id
    ).first()
    
    if not asociacion:
        raise HTTPException(
            status_code=404,
            detail="Asociación no encontrada"
        )
    
    db.delete(asociacion)
    db.commit()
    
    return {
        "mensaje": "Material eliminado de la respuesta",
        "mensaje_id": mensaje_id,
        "material_id": material_id
    }


# ============================================
# HISTORIAL COMPLETO (ACTUALIZADO)
# ============================================
@router.get("/chat/{chat_id}/historial")
def obtener_historial_chat(chat_id: int, db: Session = Depends(get_db)):
    """
    Obtiene el historial completo de un chat con materiales asociados.
    """
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat no encontrado")
    
    mensajes = (
        db.query(Mensaje)
        .filter(Mensaje.chat_id == chat_id)
        .order_by(Mensaje.fecha_mensaje.asc())
        .all()
    )
    
    historial = []
    for mensaje in mensajes:
        item = {
            "mensaje_id": mensaje.id,
            "rol": mensaje.rol,
            "tipo": mensaje.tipo,
            "fecha_mensaje": mensaje.fecha_mensaje,
            "contenido": None,
            "materiales": []
        }
        
        # Buscar si es pregunta
        pregunta = db.query(MensajePregunta).filter(
            MensajePregunta.mensaje_id == mensaje.id
        ).first()
        
        if pregunta:
            item["contenido"] = pregunta.pregunta
        else:
            # Buscar si es respuesta
            respuesta = db.query(MensajeRespuesta).filter(
                MensajeRespuesta.mensaje_id == mensaje.id
            ).first()
            
            if respuesta:
                item["contenido"] = respuesta.respuesta
                
                # ⭐ OBTENER MATERIALES ASOCIADOS
                materiales = (
                    db.query(RespuestaMaterial, MaterialEstudio,Documento)
                    .join(MaterialEstudio, RespuestaMaterial.material_id == MaterialEstudio.id)
                    .join(Documento, MaterialEstudio.documento_id == Documento.id)
                    .filter(RespuestaMaterial.mensaje_respuesta_id == respuesta.id)
                    .order_by(RespuestaMaterial.orden, RespuestaMaterial.similitud.desc())
                    .all()
                )
                
                for asoc, mat, doc in materiales:
                    item["materiales"].append({
                        "material_id": mat.id,
                        "documento_id": doc.id,
                        "nombre_documento": doc.nombre_documento,
                        "fuente": doc.fuente,
                        "url": doc.url,
                        "similitud": asoc.similitud,
                        "orden": asoc.orden
                    })
        
        historial.append(item)
    
    return {
        "chat_id": chat_id,
        "titulo": chat.titulo,
        "total_mensajes": len(historial),
        "historial": historial
    }