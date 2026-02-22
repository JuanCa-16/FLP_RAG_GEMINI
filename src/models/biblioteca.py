# src/models/biblioteca.py

from sqlalchemy import Column, Integer, String, TIMESTAMP, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.database.database import Base


class Biblioteca(Base):
    __tablename__ = "biblioteca"

    id = Column(Integer, primary_key=True, index=True)
    usuario = Column(String(15), ForeignKey("usuario.usuario", ondelete="CASCADE"), nullable=False)
    material_id = Column(Integer, ForeignKey("material_estudio.id", ondelete="CASCADE"), nullable=False)
    fecha_consulta = Column(TIMESTAMP(timezone=True), server_default=func.now())
    
    # Relaciones
    usuario_rel = relationship("Usuario", back_populates="biblioteca")
    material = relationship("MaterialEstudio", back_populates="biblioteca")