from app import db
from pgvector.sqlalchemy import Vector
from sqlalchemy import Integer, String, Text, ForeignKey
from sqlalchemy.orm import declarative_base, mapped_column, relationship
from sqlalchemy import create_engine
from app.config import SQLALCHEMY_DATABASE_URI

engine = create_engine(SQLALCHEMY_DATABASE_URI)

Base = declarative_base()


class Embeddings(Base):
    __tablename__ = "embeddings"

    id = mapped_column(Integer, primary_key=True)
    split_content = mapped_column(Text)
    embedding = mapped_column(Vector(384))
    document_id = mapped_column(ForeignKey("document.id"))


class Document(Base):
    __tablename__ = "document"

    id = mapped_column(Integer, primary_key=True)
    title = mapped_column(String)
    content = mapped_column(Text)
    children = relationship("Embeddings")


Base.metadata.create_all(engine)