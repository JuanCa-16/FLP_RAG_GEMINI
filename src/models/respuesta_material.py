# src/models/respuesta_material.py

from sqlalchemy import Column, Integer, Float, TIMESTAMP, ForeignKey, UniqueConstraint, CheckConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.database.database import Base


class RespuestaMaterial(Base):
    """
    Tabla intermedia: Relación muchos a muchos entre mensaje_respuesta y material_estudio.
    Incluye la similitud de cada material con respecto a la respuesta.
    """
    __tablename__ = "respuesta_material"

    id = Column(Integer, primary_key=True, index=True)
    mensaje_respuesta_id = Column(
        Integer, 
        ForeignKey("mensaje_respuesta.mensaje_id", ondelete="CASCADE"), 
        nullable=False
    )
    material_id = Column(
        Integer, 
        ForeignKey("material_estudio.id", ondelete="CASCADE"), 
        nullable=False
    )
    similitud = Column(Float)  # Entre 0 y 1
    orden = Column(Integer)  # Orden de recomendación
    fecha_asociacion = Column(TIMESTAMP(timezone=True), server_default=func.now())
    
    __table_args__ = (
        UniqueConstraint('mensaje_respuesta_id', 'material_id', name='uq_respuesta_material'),
        CheckConstraint('similitud >= 0 AND similitud <= 1', name='check_similitud_range'),
    )
    
    # Relaciones
    respuesta = relationship("MensajeRespuesta", back_populates="materiales_asociados")
    material = relationship("MaterialEstudio", back_populates="respuestas_asociadas")