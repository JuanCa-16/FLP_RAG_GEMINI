from sqlalchemy import Column, String, TIMESTAMP
from sqlalchemy.sql import func
from src.database.database import Base

class Usuario(Base):
    __tablename__ = "usuario"

    usuario = Column(String(15), primary_key=True)
    contrasena = Column(String(255), nullable=False)
    nombre = Column(String(20), nullable=False)
    fecha_creacion = Column(TIMESTAMP(timezone=True), server_default=func.now())
    fecha_actualizacion = Column(TIMESTAMP(timezone=True), server_default=func.now())