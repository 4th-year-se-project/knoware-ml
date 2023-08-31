from app import db
from pgvector.sqlalchemy import Vector
from sqlalchemy import Integer, String, Text
from sqlalchemy.orm import declarative_base, mapped_column
from sqlalchemy import create_engine
from app.config import SQLALCHEMY_DATABASE_URI

engine = create_engine(SQLALCHEMY_DATABASE_URI)

Base = declarative_base()


class Document(Base):
    __tablename__ = "document"

    id = mapped_column(Integer, primary_key=True)
    title = mapped_column(String)
    content = mapped_column(Text)
    embedding = mapped_column(Vector(384))


# Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)
