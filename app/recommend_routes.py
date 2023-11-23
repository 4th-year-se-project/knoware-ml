from flask import Blueprint, request

from app import models
from models import db


recomend_routes = Blueprint("recomend_routes", __name__)


@recomend_routes.route("/recommend", methods=["POST"])
def search_similar_resource():
    data = request.json
    doc_id = data.get("document_id")

    topic_id = (
        db.session.query(models.Document.topic_id)
        .filter(models.Document.id == doc_id)
        .first()
    )[0]

    course_id = (
        db.session.query(models.Topic.course_id)
        .filter(models.Topic.id == topic_id)
        .first()
    )[0]

    # print(course_id)

    similar_topic_ids = (
        db.session.query(models.Topic.id)
        .filter(models.Topic.course_id == course_id)
        .all()
    )
    similar_topic_ids = [topic_id for (topic_id,) in similar_topic_ids]

    similar_topic_docs = (
        db.session.query(models.Document.id)
        .filter(models.Document.topic_id.in_(similar_topic_ids))
        .all()
    )
    similar_topic_docs = [document_id for (document_id,) in similar_topic_docs]
    print(similar_topic_docs)

    response_dict = {}

    doc_results = (
        db.session.query(models.DocSimilarity)
        .filter(
            models.DocSimilarity.new_document_id.in_(similar_topic_docs),
            models.DocSimilarity.existing_document_id
            != models.DocSimilarity.new_document_id,
        )
        .order_by(models.DocSimilarity.similarity_score.desc())
        .slice(0, 5)
        .all()
    )

    all_users = db.session.query(models.User.id).count()

    x = 0
    for result in doc_results:
        user_count = (
            db.session.query(models.OwnsDocument)
            .filter(models.OwnsDocument.document_id == result.existing_document_id)
            .count()
        )
        recommending_doc_ratings = (
            db.session.query(models.Document.ratings)
            .filter(models.Document.id == result.existing_document_id)
            .scalar()
        )
        recommending_doc_title = (
            db.session.query(models.Document.title)
            .filter(models.Document.id == result.existing_document_id)
            .scalar()
        )
        response_dict[x] = {
            "document_id": result.existing_document_id,
            "document_title": recommending_doc_title,
            "ratings": recommending_doc_ratings,
            "similarity_score": result.similarity_score,
            "user_count": user_count,
            "similarity_weight": (user_count / all_users)
            * result.similarity_score
            * (recommending_doc_ratings / 5),
        }
        x = x + 1

    response_data = sorted(
        response_dict.values(), key=lambda x: x["similarity_weight"], reverse=True
    )

    return {"results": response_data}
