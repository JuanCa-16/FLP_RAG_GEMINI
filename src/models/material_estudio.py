
from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, Enum
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from src.database.database import Base


class MaterialEstudio(Base):
    __tablename__ = "material_estudio"

    id = Column(Integer, primary_key=True, index=True)
    fuente = Column(String(10))
    nombre_documento = Column(String(255))
    url = Column(Text)
    tematica = Column(String(100))
    competencia = Column(String(100))
    resultado_aprendizaje = Column(Text)
    nivel_dificultad = Column(String(20))
    material_embedding = Column(Vector(768))
    fecha_creacion = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now()
    )