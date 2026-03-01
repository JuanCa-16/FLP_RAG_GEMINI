from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, CheckConstraint, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.database.database import Base


class Documento(Base):
    __tablename__ = "documento"

    id = Column(Integer, primary_key=True, index=True)

    fuente = Column(String(10))
    nombre_documento = Column(String(255))
    url = Column(Text)
    tematica = Column(String(200))
    competencia = Column(String(500))
    resultado_aprendizaje = Column(Text)
    nivel_dificultad = Column(String(20))

    fecha_creacion = Column(TIMESTAMP(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint("fuente IN ('CODIGO', 'PDF', 'VIDEO', 'GIT')", name="check_documento_fuente"),
        CheckConstraint("nivel_dificultad IN ('Baja', 'Media', 'Alta')", name="check_documento_nivel"),
        UniqueConstraint('nombre_documento', 'fuente', name='uq_documento_nombre_fuente'),
    )

    # Relaciones
    materiales = relationship(
        "MaterialEstudio",
        back_populates="documento",
        cascade="all, delete-orphan"
    )

    biblioteca = relationship(
        "Biblioteca",
        back_populates="documento",
        cascade="all, delete-orphan"
    )