from sqlalchemy import Column, Integer, Text, TIMESTAMP, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from src.database.database import Base


class MaterialEstudio(Base):
    __tablename__ = "material_estudio"

    id = Column(Integer, primary_key=True)

    documento_id = Column(
        Integer,
        ForeignKey("documento.id", ondelete="CASCADE"),
        nullable=False
    )

    material_embedding = Column(Vector(768))

    # Relaciones
    documento = relationship("Documento", back_populates="materiales")

    biblioteca = relationship(
        "Biblioteca",
        back_populates="material",
    )

    respuestas_asociadas = relationship(
        "RespuestaMaterial",
        back_populates="material",
        cascade="all, delete-orphan"
    )