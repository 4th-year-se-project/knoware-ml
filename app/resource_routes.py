import os
from app import models, app
from models import db
from flask import Blueprint, jsonify, request, send_file, current_app
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import logging


# Path to the directory containing the uploaded files
pdf_directory = os.path.join(os.getcwd(), app.config["UPLOAD_FOLDER"])

modelPath = "../models/all-MiniLM-L6-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
embeddings_model = HuggingFaceEmbeddings(
    model_name=modelPath, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)


resource_routes = Blueprint("resource_routes", __name__)


@resource_routes.route("/getPdf", methods=["GET"])
def serve_pdf():
    filename = request.args.get("filename")
    pdf_file_path = os.path.join(pdf_directory, filename)
    return send_file(pdf_file_path, as_attachment=True, mimetype="application/pdf")


@resource_routes.route("/resource-info", methods=["GET"])
def get_resource_info():
    document_id = request.args.get("document_id")
    query = request.args.get("query")
    query_embedding = embeddings_model.embed_query(query)

    document = (
        db.session.query(models.Document)
        .filter(models.Document.id == document_id)
        .first()
    )

    if not document:
        return jsonify({"error": "Document not found"}), 404

    topic = (
        db.session.query(models.Topic)
        .filter(models.Topic.id == document.topic_id)
        .first()
    )

    course = (
        db.session.query(models.Course)
        .filter(models.Course.id == topic.course_id)
        .first()
    )

    embeddings = (
        db.session.query(models.Embeddings)
        .filter(models.Embeddings.document_id == document_id)
        .all()
    )

    best_embedding = ""
    best_similarity = -1

    for embedding in embeddings:
        similarity = cosine_similarity([embedding.embedding], [query_embedding])[0][0]
        if similarity > best_similarity:
            best_similarity = similarity
            best_embedding = embedding.split_content

    return (
        jsonify(
            {
                "title": document.title,
                "topic_id": document.topic_id,
                "link": document.link,
                "keywords": document.keywords,
                "content": best_embedding,
                "topics": [course.name, topic.name]
                # "topics": [course.name, topic.name, sub_topic.name],
            }
        ),
        200,
    )


@resource_routes.route("/course", methods=["GET"])
def get_course():
    document_id = request.args.get("document_id")

    # Find the document with the given document_id
    document = (
        db.session.query(models.Document)
        .filter(models.Document.id == document_id)
        .first()
    )

    if not document:
        return jsonify({"error": "Document not found"}), 404

    # Get the course associated with the document
    course = (
        db.session.query(models.Course)
        .filter(models.Course.id == document.topic.course_id)
        .first()
    )

    if not course:
        return jsonify({"error": "Course not found for the given document"}), 404

    # Construct the course data
    course_data = {
        "course_name": course.name,
        "topics": [],
    }

    topics = (
        db.session.query(models.Topic).filter(models.Topic.course_id == course.id).all()
    )

    for topic in topics:
        topic_data = {"topic_name": topic.name, "documents": []}

        documents = (
            db.session.query(models.Document)
            .filter(models.Document.topic_id == topic.id)
            .all()
        )

        for doc in documents:
            # Retrieve the similarity score from the 'doc_similarity' table
            similarity_entry = (
                db.session.query(models.DocSimilarity)
                .filter(
                    (models.DocSimilarity.new_document_id == document_id)
                    & (models.DocSimilarity.existing_document_id == doc.id)
                )
                .first()
            )
            similarity_score = (
                similarity_entry.similarity_score if similarity_entry else None
            )

            document_data = {
                "document_name": doc.title,
                "document_id": doc.id,
                "similarity_score": similarity_score,
            }

            topic_data["documents"].append(document_data)

        course_data["topics"].append(topic_data)

    return jsonify(course_data)


# Create a route to handle the DELETE request
@resource_routes.route("/resource", methods=["DELETE"])
def delete_resource():
    try:
        # Get the 'document_id' from the request's query parameters
        document_id = request.args.get("document_id")

        # Find the document with the given document_id
        document = (
            db.session.query(models.Document)
            .filter(models.Document.id == document_id)
            .first()
        )

        if not document:
            return jsonify({"error": "Document not found"}), 404

        # Delete the document from the database
        db.session.delete(document)
        db.session.commit()

        # Assuming a successful deletion, return a success message
        return jsonify(
            {"message": f"Resource with document_id {document_id} deleted successfully"}
        )

    except Exception as e:
        # Handle any errors here and return an appropriate response
        logging.exception("An error occurred during deletion:")
        return jsonify({"error": str(e)}), 500  # 500 Internal Server Error


@resource_routes.route("/topic", methods=["PUT"])
def edit_topic():
    try:
        # Get the 'document_id' and 'topic' from the request's query parameters
        document_id = request.args.get("document_id")
        topic = request.args.get("topic")
        print(topic)

        # Find the document with the given document_id
        document = (
            db.session.query(models.Document)
            .filter(models.Document.id == document_id)
            .first()
        )

        if not document:
            return jsonify({"error": "Document not found"}), 404

        # Find the corresponding topic id
        topic_id = (
            db.session.query(models.Topic).filter(models.Topic.name == topic).first().id
        )

        if not topic_id:
            return jsonify({"error": "Topic not found"}), 404

        # Update the document's topic
        document.topic_id = topic_id
        db.session.commit()

        return jsonify(
            {
                "message": f'Topic for document with ID {document_id} updated to "{topic}"'
            }
        )

    except Exception as e:
        # Handle any errors here and return an appropriate response
        return jsonify({"error": str(e)}), 500  # 500 Internal Server Error
