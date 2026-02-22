# src/models/mensaje_respuesta.py

from sqlalchemy import Column, Integer, Text, ForeignKey
from sqlalchemy.orm import relationship
from src.database.database import Base


class MensajeRespuesta(Base):
    __tablename__ = "mensaje_respuesta"

    mensaje_id = Column(Integer, ForeignKey("mensaje.id", ondelete="CASCADE"), primary_key=True)
    respuesta = Column(Text, nullable=False)
    material_id = Column(Integer, ForeignKey("material_estudio.id", ondelete="SET NULL"))
    
    # Relaciones
    mensaje = relationship("Mensaje", back_populates="respuesta")
    material = relationship("MaterialEstudio", back_populates="respuestas")