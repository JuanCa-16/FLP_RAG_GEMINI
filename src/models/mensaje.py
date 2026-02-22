# src/models/mensaje.py

from sqlalchemy import Column, Integer, String, TIMESTAMP, ForeignKey, CheckConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.database.database import Base


class Mensaje(Base):
    __tablename__ = "mensaje"

    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer, ForeignKey("chat.id", ondelete="CASCADE"), nullable=False)
    rol = Column(String(15), nullable=False)
    tipo = Column(String(10), nullable=False)
    fecha_mensaje = Column(TIMESTAMP(timezone=True), server_default=func.now())
    
    __table_args__ = (
        CheckConstraint("rol IN ('assistant', 'user')", name="check_mensaje_rol"),
        CheckConstraint("tipo IN ('CODIGO', 'PDF', 'VIDEO', 'GIT')", name="check_mensaje_tipo"),
    )
    
    # Relaciones
    chat = relationship("Chat", back_populates="mensajes")
    pregunta = relationship("MensajePregunta", uselist=False, back_populates="mensaje", cascade="all, delete-orphan") #use list para relacion uno a uno
    respuesta = relationship("MensajeRespuesta", uselist=False, back_populates="mensaje", cascade="all, delete-orphan")