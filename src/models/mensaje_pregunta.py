# src/models/mensaje_pregunta.py

from sqlalchemy import Column, Integer, Text, ForeignKey
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from src.database.database import Base


class MensajePregunta(Base):
    __tablename__ = "mensaje_pregunta"

    mensaje_id = Column(Integer, ForeignKey("mensaje.id", ondelete="CASCADE"), primary_key=True)
    pregunta = Column(Text, nullable=False)
    pregunta_embedding = Column(Vector(768))
    
    # Relaciones
    mensaje = relationship("Mensaje", back_populates="pregunta")