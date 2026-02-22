# src/models/material_estudio.py

from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, CheckConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from src.database.database import Base


class MaterialEstudio(Base):
    __tablename__ = "material_estudio"

    id = Column(Integer, primary_key=True, index=True)
    fuente = Column(String(10))
    nombre_documento = Column(String(255))
    url = Column(Text)
    tematica = Column(String(200))
    competencia = Column(String(500))
    resultado_aprendizaje = Column(Text)
    nivel_dificultad = Column(String(20))
    material_embedding = Column(Vector(768))
    fecha_creacion = Column(TIMESTAMP(timezone=True), server_default=func.now())
    
    __table_args__ = (
        CheckConstraint("fuente IN ('CODIGO', 'PDF', 'VIDEO', 'GIT')", name="check_material_fuente"),
        CheckConstraint("nivel_dificultad IN ('Baja', 'Alta', 'Media')", name="check_material_nivel"),
    )
    
    # Relaciones
    respuestas = relationship("MensajeRespuesta", back_populates="material")
    biblioteca = relationship("Biblioteca", back_populates="material", cascade="all, delete-orphan")