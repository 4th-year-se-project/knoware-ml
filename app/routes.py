from functools import wraps
import os
import time
from app import app, db, models
from flask import jsonify, request, send_file, g
from llama_hub.youtube_transcript import YoutubeTranscriptReader
from youtube_transcript_api._errors import (
    NoTranscriptFound,
)
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
from passlib.hash import sha256_crypt
import jwt
from datetime import datetime, timedelta

ffmpeg_path = '/usr/bin'
os.environ['PATH'] = f'{ffmpeg_path}:{os.environ["PATH"]}'

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

ALLOWED_EXTENSIONS = {"mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            token = auth_header.split(" ")[1] if auth_header.startswith("Bearer ") else None

        if not token:
            return jsonify({"message": "Token is missing"}), 401

        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            g.user = data['identity']  # Store the user identity in Flask's context 'g'
        except jwt.ExpiredSignatureError:
            return jsonify({"message": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"message": "Invalid token"}), 401

        return f(*args, **kwargs)

    return decorated


@app.route("/getPdf", methods=["GET"])
@token_required
def serve_pdf():
    filename = request.args.get("filename")

    user_id = (db.session.query(models.User.id).filter(models.User.username == g.user).first()[0])
    uploaded_dir = os.path.join("uploads", str(user_id))
    pdf_directory = os.path.join(os.getcwd(), uploaded_dir)
    pdf_file_path = os.path.join(pdf_directory, filename)

    return send_file(pdf_file_path, as_attachment=True, mimetype="application/pdf")


@app.route("/embed_youtube", methods=["POST"])
@token_required
def embed_youtube():
    loader = YoutubeTranscriptReader()
    user_id = (db.session.query(models.User.id).filter(models.User.username == g.user).first()[0])
    upload_dir = os.path.join("uploads", str(user_id))
    os.makedirs(upload_dir, exist_ok=True)
    data = request.json

    video_url = data.get("video_url")
    yt = YouTube(video_url)
    title = yt.title

    transcript_text = ""
    try:
        documents = loader.load_data(ytlinks=[video_url])
        transcript_text = documents[0].text
    except NoTranscriptFound as e:
        # Handle the case where no transcript is found
        stream = yt.streams.filter(only_audio=True).first()
        filename = "audio.mp3"
        filepath = os.path.join(upload_dir, filename)

        # Download the audio stream
        stream.download(output_path=upload_dir)

        # Wait for the file to be downloaded
        while not os.path.exists(
            os.path.join(upload_dir, stream.default_filename)
        ):
            time.sleep(1)

        # Rename the downloaded file to the desired filename
        os.rename(
            os.path.join(upload_dir, stream.default_filename),
            filepath,
        )

        result = whisper_model.transcribe(filepath)
        transcript_text = result["text"]
        os.remove(filepath)

    preprocessed_transcript_text = preprocess_text(transcript_text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_text(transcript_text)
    embeddings = embeddings_model.embed_documents(docs)
    keywords = get_keywords(docs)
    stored_document = models.Document(
        title=title,
        content=preprocessed_transcript_text,
        keywords=keywords,
        link=video_url,
    )
    db.session.add(stored_document)
    db.session.commit()

    # Start a new thread to execute calculate_and_store_similarity
    similarity_thread = threading.Thread(
        target=calculate_and_store_similarity,
        args=(stored_document.id, preprocessed_transcript_text),
    )
    similarity_thread.start()

    for embedding, doc in zip(embeddings, docs):
        stored_embedding = models.Embeddings(
            split_content=doc,
            embedding=embedding,
            document_id=stored_document.id,
        )
        db.session.add(stored_embedding)
    db.session.commit()

    assign_topic(stored_document)

    user_id = (db.session.query(models.User.id).filter(models.User.username == g.user).first()[0])

    #save owns document
    owns_document = models.OwnsDocument(
        user_id=user_id, document_id=stored_document.id
    )
    db.session.add(owns_document)
    db.session.commit()

    return "Embeddings saved in the database."


@app.route("/embed_pdf", methods=["POST"])
@token_required
def embed_pdf():
    loader = PDFReader()
    user_id = (db.session.query(models.User.id).filter(models.User.username == g.user).first()[0])
    upload_dir = os.path.join("uploads", str(user_id))
    os.makedirs(upload_dir, exist_ok=True)
    # Check if the post request has the file part
    if "file" not in request.files:
        return "No file part"

    file = request.files["file"]

    # If the user does not select a file, the browser may also submit an empty part without filename
    if file.filename == "":
        return "No selected file"

    if file:
        # Securely save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)

        # Load and process the PDF content
        documents = loader.load_data(
            file=Path(filepath)
        )  # Implement the PDF loading function
        document_texts = [document.text for document in documents]
        document_text = "".join(document_texts)
        preprocessed_text = preprocess_text(document_text)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150
        )
        docs = text_splitter.split_text(document_text)
        embeddings = embeddings_model.embed_documents(docs)
        keywords = get_keywords(docs)

        stored_document = models.Document(
            title=filename, content=preprocessed_text, keywords=keywords
        )
        db.session.add(stored_document)
        db.session.commit()

        # Start a new thread to execute calculate_and_store_similarity
        similarity_thread = threading.Thread(
            target=calculate_and_store_similarity,
            args=(stored_document.id, preprocessed_text),
        )
        similarity_thread.start()

        # Store the embeddings in the database
        for embedding, doc in zip(embeddings, docs):
            stored_embedding = models.Embeddings(
                split_content=doc,
                embedding=embedding,
                document_id=stored_document.id,
            )
            db.session.add(stored_embedding)
        db.session.commit()

        assign_topic(stored_document)

        user_id = (db.session.query(models.User.id).filter(models.User.username == g.user).first()[0])

        #save owns document
        owns_document = models.OwnsDocument(
            user_id=user_id, document_id=stored_document.id
        )
        db.session.add(owns_document)
        db.session.commit()

        print(user_id)

        return "Embeddings saved in the database."


@app.route("/embed_pptx", methods=["POST"])
@token_required
def embed_pptx():
    loader = PptxReader()
    user_id = (db.session.query(models.User.id).filter(models.User.username == g.user).first()[0])
    upload_dir = os.path.join("uploads", str(user_id))
    os.makedirs(upload_dir, exist_ok=True)
    # Check if the post request has the file part
    if "file" not in request.files:
        return "No file part"

    file = request.files["file"]

    # If the user does not select a file, the browser may also submit an empty part without filename
    if file.filename == "":
        return "No selected file"

    if file:
        # Securely save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)

        # Load and process the PDF content
        documents = loader.load_data(
            file=Path(filepath)
        )  # Implement the PDF loading function
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150
        )
        document_texts = [document.text for document in documents]
        document_text = "".join(document_texts)
        preprocessed_text = preprocess_text(document_text)
        docs = text_splitter.split_text(document_text)
        embeddings = embeddings_model.embed_documents(docs)
        keywords = get_keywords(docs)

        stored_document = models.Document(
            title=filename, content=preprocessed_text, keywords=keywords
        )
        db.session.add(stored_document)
        db.session.commit()

        # Start a new thread to execute calculate_and_store_similarity
        similarity_thread = threading.Thread(
            target=calculate_and_store_similarity,
            args=(stored_document.id, preprocessed_text),
        )
        similarity_thread.start()

        # Store the embeddings in the database
        for embedding, chunk in zip(embeddings, docs):
            stored_embedding = models.Embeddings(
                split_content=chunk,
                embedding=embedding,
                document_id=stored_document.id,
            )
            db.session.add(stored_embedding)
        db.session.commit()

        assign_topic(stored_document)

        #user_id = (db.session.query(models.User.id).filter(models.User.username == g.user).first()[0])

        #save owns document
        owns_document = models.OwnsDocument(
            user_id=user_id, document_id=stored_document.id
        )
        db.session.add(owns_document)
        db.session.commit()

        return "Embeddings saved in the database."


@app.route("/embed_audio", methods=["POST"])
@token_required
def embed_audio():
    user_id = (db.session.query(models.User.id).filter(models.User.username == g.user).first()[0])
    upload_dir = os.path.join("uploads", str(user_id))
    os.makedirs(upload_dir, exist_ok=True)
    # Check if the post request has the audio file part
    if "file" not in request.files:
        return "No audio file part"

    audio_file = request.files["file"]

    # If the user does not select a file, the browser may also submit an empty part without filename
    if audio_file.filename == "":
        return "No selected audio file"

    if audio_file and allowed_file(audio_file.filename):
        # Securely save the uploaded audio file
        filename = secure_filename(audio_file.filename)
        filepath = os.path.join(upload_dir, filename)
        audio_file.save(filepath)

        # Load and process the audio content
        transcript = whisper_model.transcribe(filepath)
        preprocessed_text = preprocess_text(transcript["text"])
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150
        )
        docs = text_splitter.split_text(transcript["text"])

        embeddings = embeddings_model.embed_documents(docs)

        keywords = get_keywords(docs)
        stored_document = models.Document(
            title=filename, content=preprocessed_text, keywords=keywords
        )
        db.session.add(stored_document)
        db.session.commit()

        # Start a new thread to execute calculate_and_store_similarity
        similarity_thread = threading.Thread(
            target=calculate_and_store_similarity,
            args=(stored_document.id, preprocessed_text),
        )
        similarity_thread.start()

        # Store the embeddings in the database
        for embedding, chunk in zip(embeddings, docs):
            stored_embedding = models.Embeddings(
                split_content=chunk,
                embedding=embedding,
                document_id=stored_document.id,
            )
            db.session.add(stored_embedding)
        db.session.commit()

        assign_topic(stored_document)

        user_id = (db.session.query(models.User.id).filter(models.User.username == g.user).first()[0])

        #save owns document
        owns_document = models.OwnsDocument(
            user_id=user_id, document_id=stored_document.id
        )
        db.session.add(owns_document)
        db.session.commit()

        return "Audio embeddings saved in the database."
    else:
        return "Invalid audio file format. Allowed extensions: mp3, mp4, mpeg, mpga, m4a, wav, webm"


@app.route("/embed_docx", methods=["POST"])
@token_required
def embed_docx():
    loader = DocxReader()
    user_id = (db.session.query(models.User.id).filter(models.User.username == g.user).first()[0])
    upload_dir = os.path.join("uploads", str(user_id))
    os.makedirs(upload_dir, exist_ok=True)
    # Check if the post request has the file part
    if "file" not in request.files:
        return "No file part"

    file = request.files["file"]

    # If the user does not select a file, the browser may also submit an empty part without filename
    if file.filename == "":
        return "No selected file"

    if file:
        # Securely save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)

        # Load and process the docx content
        documents = loader.load_data(
            file=Path(filepath)
        )  # Implement the docx loading function

        document_texts = [document.text for document in documents]
        document_text = "".join(document_texts)

        preprocessed_text = preprocess_text(document_text)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150
        )

        docs = text_splitter.split_text(document_text)

        embeddings = embeddings_model.embed_documents(docs)

        keywords = get_keywords(docs)

        stored_document = models.Document(
            title=filename, content=preprocessed_text, keywords=keywords
        )
        db.session.add(stored_document)
        db.session.commit()

        # Start a new thread to execute calculate_and_store_similarity
        similarity_thread = threading.Thread(
            target=calculate_and_store_similarity,
            args=(stored_document.id, preprocessed_text),
        )
        similarity_thread.start()

        # Store the embeddings in the database
        for embedding, doc in zip(embeddings, docs):
            stored_embedding = models.Embeddings(
                split_content=doc,
                embedding=embedding,
                document_id=stored_document.id,
            )
            db.session.add(stored_embedding)
        db.session.commit()

        assign_topic(stored_document)

        user_id = (db.session.query(models.User.id).filter(models.User.username == g.user).first()[0])

        #save owns document
        owns_document = models.OwnsDocument(
            user_id=user_id, document_id=stored_document.id
        )
        db.session.add(owns_document)
        db.session.commit()

        return "Embeddings saved in the database."


@app.route("/search", methods=["POST"])
@token_required
def search():
    data = request.json
    query = data.get("query")
    query_embedding = embeddings_model.embed_query(query)
    user_id = (db.session.query(models.User.id).filter(models.User.username == g.user).first()[0])

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
    results = results.join(models.OwnsDocument, models.OwnsDocument.document_id == models.Document.id)

    # Filter by user_id
    results = results.filter(
        models.OwnsDocument.user_id == user_id,
        models.Document.deleted == False
    )

    # Calculate and order by cosine distance
    results = results.order_by(
        models.Embeddings.embedding.cosine_distance(query_embedding)
    )

    doc_ids = []
    for result in results:
        embedding, document, course, topic = result
        doc_id = document.id
        # Check if this document ID is already in the results_dict
        if doc_id not in results_dict:
            doc_ids.append(doc_id)
            results_dict[doc_id] = {
                "title": document.title,
                "content": embedding.split_content,
                "doc_id": doc_id,
                "keywords": document.keywords,
                "course": course.name,
                "topic": topic.name,
            }

    # Convert the results_dict values to a list
    response_data = list(results_dict.values())[:5]

    query_log_entry = models.QueryLog(query=query, user_id=user_id, doc_ids=doc_ids)
    db.session.add(query_log_entry)
    db.session.commit()

    return {"results": response_data}, 200


@app.route("/recommend", methods=["POST"])
@token_required
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
            models.Document.deleted == False
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


@app.route("/resource-info", methods=["GET"])
@token_required
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
@token_required
def get_course():
    document_id = request.args.get("document_id")
    user_id = (db.session.query(models.User.id).filter(models.User.username == g.user).first()[0])

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
            .join(models.OwnsDocument, models.Document.id == models.OwnsDocument.document_id)
            .filter(
                models.Document.topic_id == topic.id,
                models.Document.deleted == False,
                models.OwnsDocument.user_id == user_id
            )
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
@token_required
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
        document.deleted = True
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
@token_required
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

        if len(keyword_set) >= 5:
            break

    return list(keyword_set)

def authenticate(username, password):
    user = {}
    user['username'] = username
    user['password'] = db.session.query(models.User.password).filter(models.User.username == username).first()[0]
    user['name'] = db.session.query(models.User.name).filter(models.User.username == username).first()[0]

    if sha256_crypt.verify(password, user['password']):
        return user


@app.route('/login', methods=['POST'])
def login():
    data = request.json
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({"message": "Missing username or password"}), 400

    username = data['username']
    password = data['password']

    user = authenticate(username, password)

    if user:
        token = jwt.encode({'identity': user['username'], 'exp': datetime.utcnow() + timedelta(weeks=1)}, app.config['SECRET_KEY'], algorithm='HS256')
        return jsonify({"access_token": token, "name": user["name"]}), 200
    else:
        return jsonify({"message": "Invalid username or password"}), 401


@app.route('/register', methods=["POST"])
def register():
    data = request.json
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")

    encrypted_passwored = sha256_crypt.encrypt(password)
    new_user = models.User(
        name=name, username=email, password=encrypted_passwored
    )
    db.session.add(new_user)
    db.session.commit()

    return "New user added"
  
