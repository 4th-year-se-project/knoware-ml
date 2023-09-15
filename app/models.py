from app import db
from pgvector.sqlalchemy import Vector
from sqlalchemy import ARRAY, Integer, String, Text, ForeignKey, Float
from sqlalchemy.orm import declarative_base, mapped_column, relationship
from sqlalchemy import create_engine
from app.config import SQLALCHEMY_DATABASE_URI

engine = create_engine(SQLALCHEMY_DATABASE_URI)

Base = declarative_base()


class User(Base):
    __tablename__ = "user"

    id = mapped_column(Integer, primary_key=True)
    name = mapped_column(String)


class Course(Base):
    __tablename__ = "course"

    id = mapped_column(Integer, primary_key=True)
    name = mapped_column(String)
    code = mapped_column(String)
    children = relationship("Topic")


class RegisteredTo(Base):
    __tablename__ = "registered_to"

    id = mapped_column(Integer, primary_key=True)
    user_id = mapped_column(ForeignKey("user.id"))
    course_id = mapped_column(ForeignKey("course.id"))
    user = relationship("User")
    course = relationship("Course")


class Topic(Base):
    __tablename__ = "topic"

    id = mapped_column(Integer, primary_key=True)
    name = mapped_column(String)
    course_id = mapped_column(ForeignKey("course.id"))
    embedding = mapped_column(Vector(384))
    children = relationship("SubTopic")


class SubTopic(Base):
    __tablename__ = "subtopic"

    id = mapped_column(Integer, primary_key=True)
    name = mapped_column(String)
    topic_id = mapped_column(ForeignKey("topic.id"))
    embedding = mapped_column(Vector(384))
    children = relationship("Document")


class Document(Base):
    __tablename__ = "document"

    id = mapped_column(Integer, primary_key=True)
    title = mapped_column(String)
    content = mapped_column(Text)
    subtopic_id = mapped_column(ForeignKey("subtopic.id"))
    link = mapped_column(String)
    keywords = mapped_column(ARRAY(String))
    ratings = mapped_column(Float, default=-1)


class OwnsDocument(Base):
    __tablename__ = "owns_document"

    id = mapped_column(Integer, primary_key=True)
    user_id = mapped_column(ForeignKey("user.id"))
    document_id = mapped_column(ForeignKey("document.id"))
    user = relationship("User")
    document = relationship("Document")


class DocSimilarity(Base):
    __tablename__ = "doc_similarity"

    id = mapped_column(Integer, primary_key=True)
    new_document_id = mapped_column(ForeignKey("document.id"))
    existing_document_id = mapped_column(ForeignKey("document.id"))
    similarity_score = mapped_column(Float)
    new_document = relationship("Document", foreign_keys=[new_document_id])
    existing_document = relationship("Document", foreign_keys=[existing_document_id])


class Embeddings(Base):
    __tablename__ = "embeddings"

    id = mapped_column(Integer, primary_key=True)
    split_content = mapped_column(Text)
    embedding = mapped_column(Vector(384))
    document_id = mapped_column(ForeignKey("document.id"))


Base.metadata.create_all(engine)
