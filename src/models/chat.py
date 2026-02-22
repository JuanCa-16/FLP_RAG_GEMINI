# src/models/chat.py

from sqlalchemy import Column, Integer, String, TIMESTAMP, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.database.database import Base


class Chat(Base):
    __tablename__ = "chat"

    id = Column(Integer, primary_key=True, index=True)
    usuario = Column(String(15), ForeignKey("usuario.usuario", ondelete="CASCADE"), nullable=False)
    titulo = Column(String(20))
    fecha_creacion = Column(TIMESTAMP(timezone=True), server_default=func.now())
    fecha_actualizacion = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relaciones
    propietario = relationship("Usuario", back_populates="chats") #Atributo virtual para accceder al usuario
    mensajes = relationship("Mensaje", back_populates="chat", cascade="all, delete-orphan")