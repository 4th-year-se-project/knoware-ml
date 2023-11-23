import os
from app import app, db, models
from flask import jsonify, request, send_file
from llama_hub.youtube_transcript import YoutubeTranscriptReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from werkzeug.utils import secure_filename
from llama_index import download_loader
from pathlib import Path
import whisper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
import threading
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from pytube import YouTube
import logging

modelPath = "../models/all-MiniLM-L6-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
embeddings_model = HuggingFaceEmbeddings(
    model_name=modelPath, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

PDFReader = download_loader("PDFReader")
PptxReader = download_loader("PptxReader")
DocxReader = download_loader("DocxReader")
whisper_model = whisper.load_model("small", download_root="../models/whisper")

sentence_model = SentenceTransformer(modelPath)
kw_model = KeyBERT(model=sentence_model)


# Configure the upload folder
app.config["UPLOAD_FOLDER"] = "uploads"
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

ALLOWED_EXTENSIONS = {"mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Path to the directory containing the uploaded files
pdf_directory = os.path.join(os.getcwd(), app.config["UPLOAD_FOLDER"])


@app.route("/getPdf", methods=["GET"])
def serve_pdf():
    filename = request.args.get("filename")
    pdf_file_path = os.path.join(pdf_directory, filename)
    return send_file(pdf_file_path, as_attachment=True, mimetype="application/pdf")


@app.route("/recommend", methods=["POST"])
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


@app.route("/resource-info", methods=["GET"])
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


@app.route("/course", methods=["GET"])
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
@app.route("/resource", methods=["DELETE"])
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


@app.route("/topic", methods=["PUT"])
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


def assign_topic(stored_document):
    embeddings = (
        db.session.query(models.Embeddings)
        .filter(models.Embeddings.document_id == stored_document.id)
        .all()
    )
    topics = db.session.query(models.Topic).all()

    best_topic = None
    best_similarity = -1

    # Iterate through document chunks and calculate similarities with the topics
    for embedding in embeddings:
        for topic in topics:
            similarity = cosine_similarity([embedding.embedding], [topic.embedding])[0][
                0
            ]
            if similarity > best_similarity:
                best_similarity = similarity
                best_topic = topic.id

    # Assign best matching topic to the document
    if best_topic is not None:
        stored_document.topic_id = best_topic
        db.session.commit()


def calculate_and_store_similarity(new_document_id, new_document_text):
    with app.app_context():
        existing_documents = db.session.query(models.Document).all()

        if existing_documents:
            existing_document_texts = [doc.content for doc in existing_documents]

            tfidf_vectorizer = TfidfVectorizer()

            tfidf_matrix = tfidf_vectorizer.fit_transform(existing_document_texts)
            new_document_vector = tfidf_vectorizer.transform([new_document_text])
            similarity_scores = cosine_similarity(new_document_vector, tfidf_matrix)

            self_similarity = models.DocSimilarity(
                new_document_id=new_document_id,
                existing_document_id=new_document_id,
                similarity_score=1,
            )
            db.session.add(self_similarity)

            for existing_doc, similarity_score in zip(
                existing_documents, similarity_scores[0]
            ):
                doc_similarity = models.DocSimilarity(
                    new_document_id=new_document_id,
                    existing_document_id=existing_doc.id,
                    similarity_score=similarity_score,
                )
                db.session.add(doc_similarity)

                if new_document_id != existing_doc.id:
                    reverse_doc_similarity = models.DocSimilarity(
                        new_document_id=existing_doc.id,
                        existing_document_id=new_document_id,
                        similarity_score=similarity_score,
                    )
                    db.session.add(reverse_doc_similarity)

            db.session.commit()
        else:
            print("No existing documents found. Similarity calculation skipped.")


def preprocess_text(
    text, use_stemming=True, use_lemmatization=True, remove_punctuation=True
):
    text = text.lower()
    tokens = word_tokenize(text)

    if remove_punctuation:
        tokens = [token for token in tokens if token not in string.punctuation]

    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    if use_stemming:
        processed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    elif use_lemmatization:
        processed_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    else:
        processed_tokens = filtered_tokens

    processed_tokens = [word for word in processed_tokens if word.isalpha()]

    processed_text = " ".join(processed_tokens)

    return processed_text


def get_keywords(docs):
    keyword_set = set()

    for doc in docs:
        keywords = kw_model.extract_keywords(
            doc, keyphrase_ngram_range=(1, 2), stop_words=None
        )

        for keyword, score in keywords:
            if score > 0.64:
                keyword_set.add(keyword)
            if len(keyword_set) >= 5:
                break

    return list(keyword_set)
