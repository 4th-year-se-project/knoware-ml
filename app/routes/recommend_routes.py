from flask import Blueprint, request

from app import models
from models import db


recommend_routes = Blueprint("recommend_routes", __name__)


@recommend_routes.route("/recommend", methods=["POST"])
def search_similar_resource():
    data = request.json
    doc_id = data.get("document_id")
    user_id = data.get("user_id")

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

    similar_topic_ids = (
        db.session.query(models.Topic.id)
        .filter(models.Topic.course_id == course_id)
        .all()
    )
    similar_topic_ids = [topic_id for (topic_id,) in similar_topic_ids]

    similar_topic_docs = (
        db.session.query(models.Document.id)
        .filter(
            models.Document.topic_id.in_(similar_topic_ids),
            models.Document.id != doc_id,
        )
        .all()
    )
    similar_topic_docs = [document_id for (document_id,) in similar_topic_docs]

    doc_results = (
        db.session.query(models.DocSimilarity)
        .filter(
            models.DocSimilarity.existing_document_id.in_(similar_topic_docs),
            models.DocSimilarity.new_document_id == doc_id,
            models.DocSimilarity.existing_document_id
            != models.DocSimilarity.new_document_id,
        )
        .order_by(models.DocSimilarity.similarity_score.desc())
        .all()
    )

    all_users = db.session.query(models.User.id).count()

    response_dict = []
    for count, result in enumerate(doc_results):
        document_owners = [
            owner_id[0]
            for owner_id in db.session.query(models.OwnsDocument.user_id)
            .filter(models.OwnsDocument.document_id == result.existing_document_id)
            .all()
        ]

        if user_id in document_owners:
            continue

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
        resource_topic = (
            db.session.query(models.Topic)
            .filter(
                models.Topic.id == models.Document.topic_id,
                models.Document.id == result.existing_document_id,
            )
            .first()
        )
        response_dict.append(
            {
                "document_id": result.existing_document_id,
                "document_title": recommending_doc_title,
                "topic_id": resource_topic.id,
                "topic_name": resource_topic.name,
                "ratings": recommending_doc_ratings,
                "similarity_score": result.similarity_score,
                "user_count": user_count,
                "similarity_weight": (user_count / all_users)
                * result.similarity_score
                * (recommending_doc_ratings / 5),
            }
        )

        if len(response_dict) == 5:
            print("Breaking the loop at count =", count)
            break

    return {"results": response_dict}
