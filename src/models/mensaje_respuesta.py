# src/models/mensaje_respuesta.py (ACTUALIZADO)

from sqlalchemy import Column, Integer, Text, ForeignKey
from sqlalchemy.orm import relationship
from src.database.database import Base


class MensajeRespuesta(Base):
    __tablename__ = "mensaje_respuesta"

    mensaje_id = Column(Integer, ForeignKey("mensaje.id", ondelete="CASCADE"), primary_key=True)
    respuesta = Column(Text, nullable=False)
    
    # Relaciones
    mensaje = relationship("Mensaje", back_populates="respuesta")
    
    # ⭐ NUEVA RELACIÓN: Muchos a muchos con MaterialEstudio a través de RespuestaMaterial
    materiales_asociados = relationship(
        "RespuestaMaterial", 
        back_populates="respuesta",
        cascade="all, delete-orphan"
    )
    
    # Helper property para acceder a los materiales directamente
    @property
    def materiales(self):
        """Retorna lista de materiales con su similitud"""
        return [
            {
                "material": am.material,
                "similitud": am.similitud,
                "orden": am.orden
            }
            for am in sorted(self.materiales_asociados, key=lambda x: x.orden or 0)
        ]