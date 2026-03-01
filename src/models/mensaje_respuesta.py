from sqlalchemy import Column, Integer, Text, ForeignKey, CheckConstraint
from sqlalchemy.orm import relationship
from src.database.database import Base


class MensajeRespuesta(Base):
    __tablename__ = "mensaje_respuesta"

    mensaje_id = Column(
        Integer,
        ForeignKey("mensaje.id", ondelete="CASCADE"),
        primary_key=True
    )

    respuesta = Column(Text, nullable=False)

    calificacion = Column(Integer, nullable=True)

    __table_args__ = (
        CheckConstraint("calificacion BETWEEN 1 AND 5", name="check_calificacion_rango"),
    )

    # Relaciones
    mensaje = relationship("Mensaje", back_populates="respuesta")

    materiales_asociados = relationship(
        "RespuestaMaterial",
        back_populates="respuesta",
        cascade="all, delete-orphan"
    )

    @property
    def materiales(self):
        return [
            {
                "material": am.material,
                "similitud": am.similitud,
                "orden": am.orden
            }
            for am in sorted(self.materiales_asociados, key=lambda x: x.orden or 0)
        ]