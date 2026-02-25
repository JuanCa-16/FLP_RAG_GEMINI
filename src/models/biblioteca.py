from sqlalchemy import Column, Integer, String, TIMESTAMP, ForeignKey, CheckConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.database.database import Base


class Biblioteca(Base):
    __tablename__ = "biblioteca"

    id = Column(Integer, primary_key=True, index=True)

    usuario = Column(
        String(15),
        ForeignKey("usuario.usuario", ondelete="CASCADE"),
        nullable=False
    )

    documento_id = Column(
        Integer,
        ForeignKey("documento.id", ondelete="CASCADE"),
        nullable=False
    )

    material_id = Column(
        Integer,
        ForeignKey("material_estudio.id", ondelete="CASCADE"),
        nullable=True
    )

    origen = Column(String(10), nullable=False)

    fecha_consulta = Column(TIMESTAMP(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint("origen IN ('CHAT', 'PDF', 'VIDEO')", name="check_biblioteca_origen"),
    )

    # Relaciones
    usuario_rel = relationship("Usuario", back_populates="biblioteca")

    documento = relationship("Documento", back_populates="biblioteca")

    material = relationship("MaterialEstudio", back_populates="biblioteca")