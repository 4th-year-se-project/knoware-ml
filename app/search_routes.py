from flask import Blueprint, request

from langchain.embeddings import HuggingFaceEmbeddings
from app import models
from models import db

modelPath = "../models/all-MiniLM-L6-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}

embeddings_model = HuggingFaceEmbeddings(
    model_name=modelPath, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

search_routes = Blueprint("search_routes", __name__)


@search_routes.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query")
    query_embedding = embeddings_model.embed_query(query)

    # Create a dictionary to store the results, indexed by document ID
    results_dict = {}

    # Perform a join between Embeddings, Document, Topic, and Course tables

    results = db.session.query(
        models.Embeddings, models.Document, models.Topic, models.Course
    )
    results = results.join(
        models.Document, models.Embeddings.document_id == models.Document.id
    )
    results = results.join(models.Topic, models.Document.topic_id == models.Topic.id)
    results = results.join(models.Course, models.Topic.course_id == models.Course.id)

    # Calculate and order by cosine distance
    results = results.order_by(
        models.Embeddings.embedding.cosine_distance(query_embedding)
    )

    for result in results:
        embedding, document, course, topic = result
        doc_id = document.id
        # Check if this document ID is already in the results_dict
        if doc_id not in results_dict:
            results_dict[doc_id] = {
                "title": document.title,
                "content": embedding.split_content,
                "doc_id": doc_id,
                "keywords": document.keywords,
                "course": course.name,
                "topic": topic.name,
            }

    # Convert the results_dict values to a list
    response_data = list(results_dict.values())

    return {"results": response_data}
