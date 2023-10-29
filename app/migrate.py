from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app import db
from app.models import (
    DocSimilarity,
    Document,
    Embeddings,
    User,
    Course,
    Topic
)
from langchain.embeddings import HuggingFaceEmbeddings
from app.config import SQLALCHEMY_DATABASE_URI
from data import course_data  # Import the data module

engine = create_engine(SQLALCHEMY_DATABASE_URI)
Session = sessionmaker(bind=engine)
session = Session()

modelPath = "../models/all-MiniLM-L6-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
embeddings_model = HuggingFaceEmbeddings(
    model_name=modelPath, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

try:
    # Clear all data from the tables
    session.query(Embeddings).delete()
    session.query(DocSimilarity).delete()
    session.query(Document).delete()
    session.query(Topic).delete()
    session.query(Course).delete()

    # Iterate through the list of courses in course_data
    for course_info in course_data:
        course = Course(name=course_info["name"], code=course_info["code"])
        session.add(course)

        # Iterate through the topics within each course and create Topics with Subtopics as children
        for topic_data in course_info["topics"]:
            topic = Topic(name=topic_data["name"])
            course.children.append(topic)

            session.add(topic)

            topic_embedding = embeddings_model.embed_query(topic.name)
            topic.embedding = topic_embedding

    session.commit()

except Exception as e:
    print(f"Error: {str(e)}")
    session.rollback()
finally:
    session.close()
