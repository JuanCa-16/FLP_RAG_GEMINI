import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(
    DATABASE_URL,
    connect_args={"sslmode": "require"},  # IMPORTANTE en Render
    pool_pre_ping=True,     
    pool_recycle=300 
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()

# if __name__ == "__main__":
#     try:
#         with engine.connect() as connection:
#             result = connection.execute(text("SELECT 1"))
#             print("Conexión exitosa:", result.scalar())
#     except Exception as e:
#         print("Error de conexión:", e)