from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app import db
from app.models import User, Course, Topic, SubTopic
from app.config import SQLALCHEMY_DATABASE_URI
from data import course_data  # Import the data module

engine = create_engine(SQLALCHEMY_DATABASE_URI)
Session = sessionmaker(bind=engine)
session = Session()

try:
    # Clear all data from the tables
    session.query(SubTopic).delete()
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

            # Create Subtopics and add them as children to the topic
            subtopics = [
                SubTopic(name=subtopic_name)
                for subtopic_name in topic_data["subtopics"]
            ]
            topic.children = subtopics

            session.add(topic)
            session.add_all(subtopics)

    session.commit()

except Exception as e:
    print(f"Error: {str(e)}")
    session.rollback()
finally:
    session.close()
