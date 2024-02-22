from functools import wraps
import os
import time
from app import app, db, models
from flask import jsonify, request, send_file, g, send_from_directory
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
import subprocess
from sqlalchemy import func
from collections import defaultdict
import fitz
import base64
import cv2
import requests
from io import BytesIO
from PIL import Image

# import datetime
import os
import pprint


from flask import Flask, jsonify, redirect, request, render_template, url_for
from flask_caching import Cache
from werkzeug.exceptions import Forbidden
from pylti1p3.contrib.flask import (
    FlaskOIDCLogin,
    FlaskMessageLaunch,
    FlaskRequest,
    FlaskCacheDataStorage,
)
from pylti1p3.deep_link_resource import DeepLinkResource
from pylti1p3.grade import Grade
from pylti1p3.lineitem import LineItem
from pylti1p3.tool_config import ToolConfJsonFile
from pylti1p3.registration import Registration

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

# change this to the correct path where your libreoffice is installed
uno_path = "/Applications/LibreOffice.app/Contents/MacOS"
os.environ["UNO_PATH"] = uno_path

ALLOWED_EXTENSIONS = {"mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"}


cache = Cache(app)


class ExtendedFlaskMessageLaunch(FlaskMessageLaunch):
    def validate_nonce(self):
        """
        Probably it is bug on "https://lti-ri.imsglobal.org":
        site passes invalid "nonce" value during deep links launch.
        Because of this in case of iss == http://imsglobal.org just skip nonce validation.

        """
        iss = self.get_iss()
        deep_link_launch = self.is_deep_link_launch()
        if iss == "http://imsglobal.org" and deep_link_launch:
            return self
        return super().validate_nonce()


def get_lti_config_path():
    return os.path.join(app.root_path, "configs", "moodle.json")


def get_launch_data_storage():
    return FlaskCacheDataStorage(cache)


def get_jwk_from_public_key(key_name):
    key_path = os.path.join(app.root_path, "configs", key_name)
    f = open(key_path, "r")
    key_content = f.read()
    jwk = Registration.get_jwk(key_content)
    f.close()
    return jwk


@app.route("/mlogin/", methods=["GET", "POST"])
def mlogin():
    tool_conf = ToolConfJsonFile(get_lti_config_path())
    launch_data_storage = get_launch_data_storage()

    flask_request = FlaskRequest()
    target_link_uri = flask_request.get_param("target_link_uri")
    if not target_link_uri:
        raise Exception('Missing "target_link_uri" param')

    oidc_login = FlaskOIDCLogin(
        flask_request, tool_conf, launch_data_storage=launch_data_storage
    )
    return oidc_login.enable_check_cookies().redirect(target_link_uri)


@app.route("/launch/", methods=["POST"])
def launch():
    tool_conf = ToolConfJsonFile(get_lti_config_path())
    flask_request = FlaskRequest()
    launch_data_storage = get_launch_data_storage()
    message_launch = ExtendedFlaskMessageLaunch(
        flask_request, tool_conf, launch_data_storage=launch_data_storage
    )
    message_launch_data = message_launch.get_launch_data()
    # pprint.pprint(message_launch_data.get("email"))
    email = message_launch_data.get("email")
    name = message_launch_data.get("name")
    pprint.pprint(email)
    course = message_launch_data.get(
        "https://purl.imsglobal.org/spec/lti/claim/context"
    ).get("label")
    pprint.pprint(course)

    user = db.session.query(models.User).filter(models.User.username == email).first()

    if not user:
        # add new user
        user = models.User(
            name=name,
            username=email,
            password=sha256_crypt.hash("password"),
            type="student",
        )
        db.session.add(user)
        db.session.commit()

    user_id = user.id

    course = (
        db.session.query(models.Course).filter(models.Course.code == course).first()
    )

    # check if user registered to course else add
    registered_to = (
        db.session.query(models.RegisteredTo)
        .filter(models.RegisteredTo.user_id == user_id)
        .filter(models.RegisteredTo.course_id == course.id)
        .first()
    )

    if not registered_to:
        registered_to = models.RegisteredTo(user_id=user_id, course_id=course.id)
        db.session.add(registered_to)
        db.session.commit()

    token = jwt.encode(
        {
            "identity": user.username,
            "exp": datetime.utcnow() + timedelta(weeks=1),
        },
        app.config["SECRET_KEY"],
        algorithm="HS256",
    )

    pprint.pprint(token)

    # Replace the following line with the endpoint name of your React frontend route
    react_frontend_endpoint = "http://127.0.0.1:3000/"

    # Generate the URL for the React frontend using url_for

    # Redirect to the React frontend
    return redirect(
        react_frontend_endpoint + "?" + "token=" + token + "&" + "name=" + name
    )
    # set to token
    # return render_template("game.html", **tpl_kwargs)


@app.route("/jwks/", methods=["GET"])
def get_jwks():
    tool_conf = ToolConfJsonFile(get_lti_config_path())
    return jsonify({"keys": tool_conf.get_jwks()})


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        if "Authorization" in request.headers:
            auth_header = request.headers["Authorization"]
            token = (
                auth_header.split(" ")[1] if auth_header.startswith("Bearer ") else None
            )

        if not token:
            return jsonify({"message": "Token is missing"}), 401

        try:
            data = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
            g.user = data["identity"]  # Store the user identity in Flask's context 'g'
        except jwt.ExpiredSignatureError:
            return jsonify({"message": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"message": "Invalid token"}), 401

        return f(*args, **kwargs)

    return decorated


@app.route("/getmedia/<int:doc_id>/<path:filename>", methods=["GET", "OPTIONS"])
def serve_media(doc_id, filename):
    # doc_id = request.args.get('doc_id')
    if doc_id is None:
        return "Missing 'doc_id' parameter", 400

    user_id = (
        db.session.query(models.OwnsDocument.user_id)
        .filter(models.OwnsDocument.document_id == doc_id)
        .first()[0]
    )
    uploaded_dir = os.path.join("uploads", str(user_id))
    print(filename)
    return send_from_directory(uploaded_dir, filename)


@app.route("/getPdf", methods=["GET"])
def serve_pdf():
    doc_id = request.args.get("document_id")
    filename = (
        db.session.query(models.Document.title)
        .filter(models.Document.id == doc_id)
        .first()[0]
    )

    owner_id = (
        db.session.query(models.OwnsDocument.user_id)
        .filter(models.OwnsDocument.document_id == doc_id)
        .first()[0]
    )
    uploaded_dir = os.path.join("uploads", str(owner_id))
    pdf_directory = os.path.join(os.getcwd(), uploaded_dir)

    if not filename.lower().endswith(".pdf"):
        filename = os.path.splitext(filename)[0] + ".pdf"

    pdf_file_path = os.path.join(pdf_directory, filename)

    return send_file(pdf_file_path, as_attachment=True, mimetype="application/pdf")


@app.route("/embed_youtube", methods=["POST"])
@token_required
def embed_youtube():
    loader = YoutubeTranscriptReader()
    user_id = (
        db.session.query(models.User.id)
        .filter(models.User.username == g.user)
        .first()[0]
    )
    upload_dir = os.path.join("uploads", str(user_id))
    os.makedirs(upload_dir, exist_ok=True)
    data = request.json

    video_url = data.get("video_url")
    yt = YouTube(video_url)
    title = yt.title
    yt_length = yt.length

    thumbnail_url = yt.thumbnail_url
    response = requests.get(thumbnail_url)
    image = Image.open(BytesIO(response.content))

    filename = title + ".jpg"
    filepath = os.path.join(upload_dir, filename)
    image.save(filepath)

    transcript_text = ""
    try:
        documents = loader.load_data(ytlinks=[video_url])
        print(documents)
        transcript_text = documents[0].text
    except NoTranscriptFound as e:
        # Handle the case where no transcript is found
        stream = yt.streams.filter(only_audio=True).first()
        filename = "audio.mp3"
        filepath = os.path.join(upload_dir, filename)

        # Download the audio stream
        stream.download(output_path=upload_dir)

        # Wait for the file to be downloaded
        while not os.path.exists(os.path.join(upload_dir, stream.default_filename)):
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
    if existsDuplicate(preprocessed_transcript_text, user_id):
        return "Duplicate document", 400

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_text(transcript_text)
    embeddings = embeddings_model.embed_documents(docs)

    num_embeddings = len(embeddings)
    time_interval = yt_length / num_embeddings
    timestamps = [i * time_interval for i in range(num_embeddings)]

    keywords = get_keywords(docs)
    stored_document = models.Document(
        title=title,
        content=preprocessed_transcript_text,
        keywords=keywords,
        link=video_url,
        type="youtube",
    )
    db.session.add(stored_document)
    db.session.commit()

    # Start a new thread to execute calculate_and_store_similarity
    similarity_thread = threading.Thread(
        target=calculate_and_store_similarity,
        args=(stored_document.id, preprocessed_transcript_text),
    )
    similarity_thread.start()

    for embedding, doc, timestamp in zip(embeddings, docs, timestamps):
        stored_embedding = models.Embeddings(
            split_content=doc,
            embedding=embedding,
            document_id=stored_document.id,
            timestamp=timestamp,
        )
        db.session.add(stored_embedding)
    db.session.commit()

    assign_topic(stored_document)

    user_id = (
        db.session.query(models.User.id)
        .filter(models.User.username == g.user)
        .first()[0]
    )

    # save owns document
    owns_document = models.OwnsDocument(user_id=user_id, document_id=stored_document.id)
    db.session.add(owns_document)
    db.session.commit()

    return "Embeddings saved in the database."


@app.route("/embed_pdf", methods=["POST"])
@token_required
def embed_pdf():
    loader = PDFReader()
    user_id = (
        db.session.query(models.User.id)
        .filter(models.User.username == g.user)
        .first()[0]
    )
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
        documents = loader.load_data(file=Path(filepath))
        document_texts = [document.text for document in documents]
        pages = [document.metadata["page_label"] for document in documents]
        document_text = "".join(document_texts)
        preprocessed_text = preprocess_text(document_text)
        if existsDuplicate(preprocessed_text, user_id):
            return "Duplicate document", 400
        embeddings = embeddings_model.embed_documents(document_texts)
        keywords = get_keywords(document_texts)

        stored_document = models.Document(
            title=filename, content=preprocessed_text, keywords=keywords, type="pdf"
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
        for embedding, doc, page in zip(embeddings, document_texts, pages):
            stored_embedding = models.Embeddings(
                split_content=doc,
                embedding=embedding,
                document_id=stored_document.id,
                page=page,
            )
            db.session.add(stored_embedding)
        db.session.commit()

        assign_topic(stored_document)

        user_id = (
            db.session.query(models.User.id)
            .filter(models.User.username == g.user)
            .first()[0]
        )

        # save owns document
        owns_document = models.OwnsDocument(
            user_id=user_id, document_id=stored_document.id
        )
        db.session.add(owns_document)
        db.session.commit()

        filename_without_extension = filename.split(".")[0] + ".png"

        preview_path = os.path.join(upload_dir, filename_without_extension)
        convert_pdf_page_to_image(filepath, 1, preview_path)

        return "Embeddings saved in the database."


def convert_to_pdf(input_file, output_file):
    try:
        subprocess.run(
            ["unoconv", "-f", "pdf", "-o", output_file, input_file], check=True
        )
        print(f"Conversion successful: {input_file} -> {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")


@app.route("/embed_pptx", methods=["POST"])
@token_required
def embed_pptx():
    loader = PptxReader()
    user_id = (
        db.session.query(models.User.id)
        .filter(models.User.username == g.user)
        .first()[0]
    )
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
        original_filepath = os.path.join(upload_dir, filename)
        file.save(original_filepath)

        # Convert to PDF
        pdf_filename = os.path.splitext(filename)[0] + ".pdf"
        pdf_filepath = os.path.join(upload_dir, pdf_filename)
        convert_to_pdf(original_filepath, pdf_filepath)

        # Load and process the PDF content
        documents = loader.load_data(
            file=Path(original_filepath)
        )  # Implement the PDF loading function
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150
        )
        document_texts = [document.text for document in documents]
        document_text = "".join(document_texts)
        preprocessed_text = preprocess_text(document_text)
        if existsDuplicate(preprocessed_text, user_id):
            return "Duplicate document", 400
        docs = text_splitter.split_text(document_text)
        embeddings = embeddings_model.embed_documents(docs)
        keywords = get_keywords(docs)

        stored_document = models.Document(
            title=filename, content=preprocessed_text, keywords=keywords, type="ppt"
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

        # user_id = (db.session.query(models.User.id).filter(models.User.username == g.user).first()[0])

        # save owns document
        owns_document = models.OwnsDocument(
            user_id=user_id, document_id=stored_document.id
        )
        db.session.add(owns_document)
        db.session.commit()

        os.remove(original_filepath)

        return "Embeddings saved in the database."


@app.route("/embed_audio", methods=["POST"])
@token_required
def embed_audio():
    user_id = (
        db.session.query(models.User.id)
        .filter(models.User.username == g.user)
        .first()[0]
    )
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
        if existsDuplicate(preprocessed_text, user_id):
            return "Duplicate document", 400

        # Desired time length for each combined segment in seconds
        desired_combined_segment_length = 30.0

        # Combine consecutive segments until the desired length is reached
        docs = []
        current_doc = {"text": "", "start": 0.0, "end": 0.0}

        for segment in transcript["segments"]:
            if (
                segment["start"] - current_doc["start"]
                <= desired_combined_segment_length
            ):
                # Concatenate segments
                current_doc["text"] += " " + segment["text"]
                current_doc["end"] = segment["end"]
            else:
                # Check if the combined segment meets the desired length
                if (
                    current_doc["end"] - current_doc["start"]
                    >= desired_combined_segment_length
                ):
                    docs.append(current_doc)

                # Start a new combined segment
                current_doc = {
                    "text": segment["text"],
                    "start": segment["start"],
                    "end": segment["end"],
                }

        # Check and add the last combined segment
        if current_doc["text"]:
            docs.append(current_doc)

        embeddings = embeddings_model.embed_documents([doc["text"] for doc in docs])

        keywords = get_keywords([doc["text"] for doc in docs])
        stored_document = models.Document(
            title=filename, content=preprocessed_text, keywords=keywords, type="audio"
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
                split_content=chunk["text"],
                embedding=embedding,
                document_id=stored_document.id,
                timestamp=chunk["start"],
            )
            db.session.add(stored_embedding)
        db.session.commit()

        assign_topic(stored_document)

        user_id = (
            db.session.query(models.User.id)
            .filter(models.User.username == g.user)
            .first()[0]
        )

        # save owns document
        owns_document = models.OwnsDocument(
            user_id=user_id, document_id=stored_document.id
        )
        db.session.add(owns_document)
        db.session.commit()

        base_name, extension = os.path.splitext(filename)
        if extension == ".mp4" or extension == ".mpeg" or extension == ".webm":
            filepath = os.path.join(upload_dir, filename)
            new_extension = ".jpg"

            new_filepath = os.path.join(upload_dir, base_name + new_extension)

            video_capture = cv2.VideoCapture(filepath)
            video_capture.set(cv2.CAP_PROP_POS_MSEC, 0)
            success, image = video_capture.read()

            if success:
                cv2.imwrite(new_filepath, image)

        return "Audio embeddings saved in the database."
    else:
        return "Invalid audio file format. Allowed extensions: mp3, mp4, mpeg, mpga, m4a, wav, webm"


@app.route("/embed_docx", methods=["POST"])
@token_required
def embed_docx():
    loader = DocxReader()
    user_id = (
        db.session.query(models.User.id)
        .filter(models.User.username == g.user)
        .first()[0]
    )
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
        original_filepath = os.path.join(upload_dir, filename)
        file.save(original_filepath)

        print(f"File extension: {os.path.splitext(filename)[-1]}")

        # Convert to PDF
        pdf_filename = os.path.splitext(filename)[0] + ".pdf"
        pdf_filepath = os.path.join(upload_dir, pdf_filename)
        convert_to_pdf(original_filepath, pdf_filepath)

        # Load and process the docx content
        documents = loader.load_data(
            file=Path(original_filepath)
        )  # Implement the docx loading function

        document_texts = [document.text for document in documents]
        document_text = "".join(document_texts)

        preprocessed_text = preprocess_text(document_text)
        if existsDuplicate(preprocessed_text, user_id):
            return "Duplicate document", 400
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150
        )

        docs = text_splitter.split_text(document_text)

        embeddings = embeddings_model.embed_documents(docs)

        keywords = get_keywords(docs)

        stored_document = models.Document(
            title=filename, content=preprocessed_text, keywords=keywords, type="doc"
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

        user_id = (
            db.session.query(models.User.id)
            .filter(models.User.username == g.user)
            .first()[0]
        )

        # save owns document
        owns_document = models.OwnsDocument(
            user_id=user_id, document_id=stored_document.id
        )
        db.session.add(owns_document)
        db.session.commit()

        os.remove(original_filepath)

        return "Embeddings saved in the database."


@app.route("/search", methods=["POST"])
@token_required
def search():
    data = request.json
    query = data.get("query")
    filter_file_format = data.get("file_format")
    filter_date = data.get("date")
    filter_course = data.get("course")
    filter_label = data.get("label")

    query_embedding = embeddings_model.embed_query(query)
    user_id = (
        db.session.query(models.User.id)
        .filter(models.User.username == g.user)
        .first()[0]
    )
    upload_dir = os.path.join("uploads", str(user_id))

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
    results = results.join(
        models.OwnsDocument, models.OwnsDocument.document_id == models.Document.id
    )

    # Filter by user_id
    results = results.filter(
        models.OwnsDocument.user_id == user_id, models.Document.deleted == False
    )

    # Calculate and order by cosine distance
    results = results.order_by(
        models.Embeddings.embedding.cosine_distance(query_embedding)
    )

    if filter_file_format:
        results = results.filter(models.Document.type == filter_file_format)

    if filter_label:
        results = results.filter(models.Document.label == filter_label)

    if filter_date:
        current_date = datetime.now()

        if filter_date == "1 day ago":
            results = results.filter(
                models.Document.date_created >= current_date - timedelta(days=1)
            )

        if filter_date == "2 days ago":
            results = results.filter(
                models.Document.date_created >= current_date - timedelta(days=2)
            )

        if filter_date == "1 week ago":
            results = results.filter(
                models.Document.date_created >= current_date - timedelta(weeks=1)
            )

        if filter_date == "1 month ago":
            results = results.filter(
                models.Document.date_created >= current_date - timedelta(weeks=2)
            )

    doc_ids = []
    for result in results:
        embedding, document, course, topic = result
        doc_id = document.id
        base64_image = None

        # Check if this document ID is already in the results_dict
        if doc_id not in results_dict:
            if filter_course:
                if topic.name != filter_course:
                    continue

            doc_ids.append(doc_id)
            if embedding.timestamp:
                timestamp = timedelta(seconds=float(embedding.timestamp))

                hours, remainder = divmod(timestamp.seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                formatted_timestamp = "{:02}:{:02}:{:02}".format(
                    int(hours), int(minutes), int(seconds)
                )

                base_name, extension = os.path.splitext(document.title)

                if extension == ".mp4" or extension == ".mpeg" or extension == ".webm":
                    filepath = os.path.join(upload_dir, document.title)
                    new_extension = ".jpg"

                    new_file_name = f"{base_name}-{formatted_timestamp}"
                    new_filepath = os.path.join(
                        upload_dir, new_file_name + new_extension
                    )

                    time = embedding.timestamp * 1000

                    video_capture = cv2.VideoCapture(filepath)
                    video_capture.set(cv2.CAP_PROP_POS_MSEC, time)
                    success, image = video_capture.read()

                    if success:
                        cv2.imwrite(new_filepath, image)
                        with open(new_filepath, "rb") as image_file:
                            base64_image = base64.b64encode(image_file.read()).decode(
                                "utf-8"
                            )

            else:
                formatted_timestamp = None

            if (
                document.type == "pdf"
                or document.type == "ppt"
                or document.type == "doc"
            ):
                filepath = os.path.join(upload_dir, document.title)
                filename_without_extension = (
                    document.title.split(".")[0] + "-" + str(embedding.page) + ".png"
                )
                preview_path = os.path.join(upload_dir, filename_without_extension)
                convert_pdf_page_to_image(filepath, embedding.page, preview_path)

                with open(preview_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

            if document.type == "youtube":
                filename = document.title + ".jpg"
                filepath = os.path.join(upload_dir, filename)

                with open(filepath, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

            results_dict[doc_id] = {
                "title": document.title,
                "type": document.type,
                "content": embedding.split_content,
                "doc_id": doc_id,
                "embedding_id": embedding.id,
                "keywords": document.keywords,
                "course": course.name,
                "topic": topic.name,
                "page": embedding.page,
                "page_image": base64_image,
                "timestamp": formatted_timestamp,
                "isRecommended": False,
                "label": document.label
            }

    # Convert the results_dict values to a list
    response_data = list(results_dict.values())[:5]

    query_log_entry = models.QueryLog(query=query, user_id=user_id, doc_ids=doc_ids)
    db.session.add(query_log_entry)
    db.session.commit()

    return {"results": response_data}, 200


@app.route("/search-recommend", methods=["POST"])
@token_required
def search_recommendation():
    data = request.json
    query = data.get("query")

    query_embedding = embeddings_model.embed_query(query)
    user_id = (
        db.session.query(models.User.id)
        .filter(models.User.username == g.user)
        .first()[0]
    )

    results_dict = {}

    results = db.session.query(
        models.Embeddings,
        models.Document,
        models.Topic,
        models.Course,
        models.OwnsDocument,
    )
    results = results.join(
        models.Document, models.Embeddings.document_id == models.Document.id
    )
    results = results.join(models.Topic, models.Document.topic_id == models.Topic.id)
    results = results.join(models.Course, models.Topic.course_id == models.Course.id)
    results = results.join(
        models.OwnsDocument, models.OwnsDocument.document_id == models.Document.id
    )

    # Filter others resources
    results = results.filter(
        models.OwnsDocument.user_id != user_id, models.Document.deleted == False
    )

    # Calculate and order by cosine distance
    results = results.order_by(
        models.Embeddings.embedding.cosine_distance(query_embedding)
    )

    print(results)

    doc_ids = []
    for result in results:
        embedding, document, course, topic, owns_document = result
        doc_id = document.id
        base64_image = None
        id = owns_document.user_id

        upload_dir = os.path.join("uploads", str(id))

        # Check if this document ID is already in the results_dict
        if doc_id not in results_dict:
            doc_ids.append(doc_id)
            if embedding.timestamp:
                timestamp = timedelta(seconds=float(embedding.timestamp))

                hours, remainder = divmod(timestamp.seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                formatted_timestamp = "{:02}:{:02}:{:02}".format(
                    int(hours), int(minutes), int(seconds)
                )

                base_name, extension = os.path.splitext(document.title)

                if extension == ".mp4" or extension == ".mpeg" or extension == ".webm":
                    filepath = os.path.join(upload_dir, document.title)
                    new_extension = ".jpg"

                    new_file_name = f"{base_name}-{formatted_timestamp}"
                    new_filepath = os.path.join(
                        upload_dir, new_file_name + new_extension
                    )

                    time = embedding.timestamp * 1000

                    video_capture = cv2.VideoCapture(filepath)
                    video_capture.set(cv2.CAP_PROP_POS_MSEC, time)
                    success, image = video_capture.read()

                    if success:
                        cv2.imwrite(new_filepath, image)
                        with open(new_filepath, "rb") as image_file:
                            base64_image = base64.b64encode(image_file.read()).decode(
                                "utf-8"
                            )

            else:
                formatted_timestamp = None

            if (
                document.type == "pdf"
                or document.type == "ppt"
                or document.type == "doc"
            ):
                filepath = os.path.join(upload_dir, document.title)
                filename_without_extension = (
                    document.title.split(".")[0] + "-" + str(embedding.page) + ".png"
                )
                preview_path = os.path.join(upload_dir, filename_without_extension)
                convert_pdf_page_to_image(filepath, embedding.page, preview_path)

                with open(preview_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

            if document.type == "youtube":
                filename = document.title + ".jpg"
                filepath = os.path.join(upload_dir, filename)

                with open(filepath, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

            results_dict[doc_id] = {
                "title": document.title,
                "type": document.type,
                "content": embedding.split_content,
                "doc_id": doc_id,
                "keywords": document.keywords,
                "course": course.name,
                "topic": topic.name,
                "page": embedding.page,
                "page_image": base64_image,
                "timestamp": formatted_timestamp,
                "isRecommeded": True
            }

    # Convert the results_dict values to a list
    response_data = list(results_dict.values())[:5]

    return {"results": response_data}, 200


@app.route("/recommend", methods=["POST"])
@token_required
def search_similar_resource():
    data = request.json
    doc_ids = data.get("document_ids")
    user_id = (
        db.session.query(models.User.id)
        .filter(models.User.username == g.user)
        .first()[0]
    )

    doc_results = (
        db.session.query(models.DocSimilarity)
        .filter(
            models.DocSimilarity.new_document_id.in_(doc_ids),
            models.DocSimilarity.existing_document_id
            != models.DocSimilarity.new_document_id,
        )
        .order_by(models.DocSimilarity.similarity_score.desc())
        .all()
    )

    all_users = db.session.query(models.User.id).count()

    results_dict = {}

    for result in doc_results:
        existing_doc_id = result.existing_document_id
        base64_image = None

        document_owners = [
            owner_id[0]
            for owner_id in db.session.query(models.OwnsDocument.user_id)
            .filter(models.OwnsDocument.document_id == result.existing_document_id)
            .limit(5)
        ]

        if user_id in document_owners:
            continue

        document_owner_id = (
            db.session.query(models.OwnsDocument.user_id)
            .filter(
                models.OwnsDocument.user_id != user_id,
                models.OwnsDocument.document_id == result.existing_document_id,
            )
            .first()
        )[0]

        document_content = (
            db.session.query(models.Document.content)
            .filter(models.Document.id == result.existing_document_id)
            .first()
        )[0]

        is_duplicate = existsDuplicate(document_content, user_id)

        if is_duplicate:
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
        recommending_doc_type = (
            db.session.query(models.Document.type)
            .filter(models.Document.id == result.existing_document_id)
            .first()
        )[0]
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
        embedding = (
            (db.session.query(models.Embeddings))
            .filter(models.Embeddings.document_id == existing_doc_id)
            .first()
        )
        course_id = (
            db.session.query(models.Topic.course_id)
            .filter(models.Topic.id == resource_topic.id)
            .first()[0]
        )
        course_name = (
            db.session.query(models.Course.name)
            .filter(models.Course.id == course_id)
            .first()[0]
        )

        upload_dir = os.path.join("uploads", str(document_owner_id))

        if embedding.timestamp:
            timestamp = timedelta(seconds=float(embedding.timestamp))

            hours, remainder = divmod(timestamp.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            formatted_timestamp = "{:02}:{:02}:{:02}".format(
                int(hours), int(minutes), int(seconds)
            )

            base_name, extension = os.path.splitext(recommending_doc_title)

            if extension == ".mp4" or extension == ".mpeg" or extension == ".webm":
                filepath = os.path.join(upload_dir, recommending_doc_title)
                new_extension = ".jpg"

                new_file_name = f"{base_name}-{formatted_timestamp}"
                new_filepath = os.path.join(upload_dir, new_file_name + new_extension)

                time = embedding.timestamp * 1000

                video_capture = cv2.VideoCapture(filepath)
                video_capture.set(cv2.CAP_PROP_POS_MSEC, time)
                success, image = video_capture.read()

                if success:
                    cv2.imwrite(new_filepath, image)
                    with open(new_filepath, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode(
                            "utf-8"
                        )

        else:
            formatted_timestamp = None

        if (
            recommending_doc_type == "pdf"
            or recommending_doc_type == "ppt"
            or recommending_doc_type == "doc"
        ):
            filename = recommending_doc_title.split(".")[0] + ".png"
            preview_path = os.path.join(upload_dir, filename)

            with open(preview_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        if recommending_doc_type == "youtube":
            filename = recommending_doc_title + ".jpg"
            filepath = os.path.join(upload_dir, filename)

            with open(filepath, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        results_dict[existing_doc_id] = {
            "title": recommending_doc_title,
            "type": recommending_doc_type,
            "content": embedding.split_content,
            "document_owner": document_owners[0],
            "doc_id": result.existing_document_id,
            "course": course_name,
            "topic": resource_topic.name,
            "page": embedding.page,
            "page_image": base64_image,
            "timestamp": formatted_timestamp,
            "ratings": recommending_doc_ratings,
            "similarity_score": result.similarity_score,
            "user_count": user_count,
            "similarity_weight": (user_count / all_users)
            * result.similarity_score
            * (recommending_doc_ratings / 5),
            "keywords":"",
            "isRecommended": True
        }

    response_data = list(results_dict.values())[:5]

    return {"results": response_data}, 200


@app.route("/resource-info", methods=["GET"])
@token_required
def get_resource_info():
    document_id = request.args.get("document_id")
    query = request.args.get("query")
    query_embedding = embeddings_model.embed_query(query)
    user_id = (
        db.session.query(models.User.id)
        .filter(models.User.username == g.user)
        .first()[0]
    )
    upload_dir = os.path.join("uploads", str(user_id))

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
    base64_image = None
    file_url = None

    for embedding in embeddings:
        similarity = cosine_similarity([embedding.embedding], [query_embedding])[0][0]
        if similarity > best_similarity:
            best_similarity = similarity
            best_embedding = embedding

    if document.type == "pdf" or document.type == "ppt" or document.type == "doc":
        filepath = os.path.join(upload_dir, document.title)
        filename_without_extension = (
            document.title.split(".")[0] + "-" + str(embedding.page) + ".png"
        )
        preview_path = os.path.join(upload_dir, filename_without_extension)
        convert_pdf_page_to_image(filepath, embedding.page, preview_path)

        with open(preview_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        # file_url = f'/getmedia/{user_id}/{document.title}?timestamp={embedding.timestamp}'

    if document.type == "youtube":
        filename = document.title + ".jpg"
        filepath = os.path.join(upload_dir, filename)

        with open(filepath, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    if best_embedding.timestamp:
        timestamp = timedelta(seconds=best_embedding.timestamp)

        hours, remainder = divmod(timestamp.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_timestamp = "{:02}:{:02}:{:02}".format(
            int(hours), int(minutes), int(seconds)
        )

        base_name, extension = os.path.splitext(document.title)
        if extension == ".mp4" or extension == ".mpeg" or extension == ".webm":
            filepath = os.path.join(upload_dir, document.title)
            new_extension = ".jpg"

            new_file_name = f"{base_name}-{formatted_timestamp}"
            new_filepath = os.path.join(upload_dir, new_file_name + new_extension)

            time = best_embedding.timestamp * 1000

            video_capture = cv2.VideoCapture(filepath)
            video_capture.set(cv2.CAP_PROP_POS_MSEC, time)
            success, image = video_capture.read()

            if success:
                cv2.imwrite(new_filepath, image)
                with open(new_filepath, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        url = "http://localhost:8081"
        file_url = f"{url}/getmedia/{document.id}/{document.title}"
        print(file_url)

    else:
        formatted_timestamp = None

    return (
        jsonify(
            {
                "title": document.title,
                "topic_id": document.topic_id,
                "link": document.link,
                "keywords": document.keywords,
                "content": best_embedding.split_content,
                "topics": [course.name, topic.name],
                "page": best_embedding.page,
                "page_image": base64_image,
                "timestamp": formatted_timestamp,
                "type": document.type,
                "file_url": file_url,
                "label": document.label,
                # "topics": [course.name, topic.name, sub_topic.name],
            }
        ),
        200,
    )


@app.route("/get_all_resources", methods=["GET"])
@token_required
def get_all_resources():
    user_id = (
        db.session.query(models.User.id)
        .filter(models.User.username == g.user)
        .first()[0]
    )
    upload_dir = os.path.join("uploads", str(user_id))

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
    results = results.join(
        models.OwnsDocument, models.OwnsDocument.document_id == models.Document.id
    )

    # Filter by user_id
    results = results.filter(
        models.OwnsDocument.user_id == user_id, models.Document.deleted == False
    )

    doc_ids = []
    for result in results:
        embedding, document, course, topic = result
        doc_id = document.id
        base64_image = None

        # Check if this document ID is already in the results_dict
        if doc_id not in results_dict:
            doc_ids.append(doc_id)
            if (
                document.type == "pdf"
                or document.type == "ppt"
                or document.type == "doc"
            ):
                filename = document.title.split(".")[0] + ".png"
                preview_path = os.path.join(upload_dir, filename)

                with open(preview_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

            if document.type == "youtube":
                filename = document.title + ".jpg"
                filepath = os.path.join(upload_dir, filename)

                with open(filepath, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

            if embedding.timestamp:
                timestamp = timedelta(seconds=embedding.timestamp)

                hours, remainder = divmod(timestamp.seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                formatted_timestamp = "{:02}:{:02}:{:02}".format(
                    int(hours), int(minutes), int(seconds)
                )

                base_name, extension = os.path.splitext(document.title)
                if extension == ".mp4" or extension == ".mpeg" or extension == ".webm":
                    new_extension = ".jpg"
                    preview_path = os.path.join(upload_dir, base_name + new_extension)
                    with open(preview_path, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode(
                            "utf-8"
                        )

            else:
                formatted_timestamp = None

            results_dict[doc_id] = {
                "title": document.title,
                "type": document.type,
                "content": embedding.split_content,
                "doc_id": doc_id,
                "embedding_id": embedding.id,
                "keywords": document.keywords,
                "course": course.name,
                "topic": topic.name,
                "page": embedding.page,
                "page_image": base64_image,
                "timestamp": formatted_timestamp,
                "label": document.label,
                "link": document.link,
                "isRecommended": False
            }

    # Convert the results_dict values to a list
    response_data = list(results_dict.values())

    return jsonify({"results": response_data}), 200


@app.route("/course", methods=["GET"])
@token_required
def get_course():
    document_id = request.args.get("document_id")
    user_id = (
        db.session.query(models.User.id)
        .filter(models.User.username == g.user)
        .first()[0]
    )

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
            .join(
                models.OwnsDocument,
                models.Document.id == models.OwnsDocument.document_id,
            )
            .filter(
                models.Document.topic_id == topic.id,
                models.Document.deleted == False,
                models.OwnsDocument.user_id == user_id,
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


@app.route("/courses", methods=["GET"])
def get_courses():
    courses = db.session.query(models.Course).all()

    # courses = models.Course.query.all()
    course_data = []

    for course in courses:
        topic_data = []
        topics = (
            db.session.query(models.Topic)
            .filter(models.Topic.course_id == course.id)
            .all()
        )
        for topic in topics:
            topic_data.append(topic.name)

        course_data.append(
            {
                "id": course.id,
                "courseName": course.name,
                "courseCode": course.code,
                "topics": topic_data,
            }
        )

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

    user_id = (
        db.session.query(models.User.id)
        .filter(models.User.username == g.user)
        .first()[0]
    )
    print(user_id)
    # get topics from registered courses
    registered_courses = (
        db.session.query(models.Course)
        .join(
            models.RegisteredTo,
            models.Course.id == models.RegisteredTo.course_id,
        )
        .filter(
            models.RegisteredTo.user_id == user_id,
        )
        .all()
    )
    candidate_topics = []
    for course in registered_courses:
        topics = (
            db.session.query(models.Topic)
            .filter(models.Topic.course_id == course.id)
            .all()
        )
        candidate_topics.extend(topics)

    # topics = db.session.query(models.Topic).all()

    best_topic = None
    best_similarity = -1

    # Iterate through document chunks and calculate similarities with the topics
    for embedding in embeddings:
        for topic in candidate_topics:
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


def convert_pdf_page_to_image(pdf_path, page_number, output_image_path):
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    # Get the specified page
    pdf_page = pdf_document.load_page(page_number - 1)

    # Create an image of the page
    image = pdf_page.get_pixmap()

    # Save the image to a file
    image.save(output_image_path)

    # Close the PDF document
    pdf_document.close()


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        if "Authorization" in request.headers:
            auth_header = request.headers["Authorization"]
            token = (
                auth_header.split(" ")[1] if auth_header.startswith("Bearer ") else None
            )

        if not token:
            return jsonify({"message": "Token is missing"}), 401

        try:
            data = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
            g.user = data["identity"]  # Store the user identity in Flask's context 'g'
        except jwt.ExpiredSignatureError:
            return jsonify({"message": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"message": "Invalid token"}), 401

        return f(*args, **kwargs)

    return decorated


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
    user["username"] = username
    user["password"] = (
        db.session.query(models.User.password)
        .filter(models.User.username == username)
        .first()[0]
    )
    user["name"] = (
        db.session.query(models.User.name)
        .filter(models.User.username == username)
        .first()[0]
    )

    if sha256_crypt.verify(password, user["password"]):
        return user


def existsDuplicate(content, user_id):
    documents = db.session.query(models.Document).all()
    for document in documents:
        if document.content == content:
            owns_documents = db.session.query(models.OwnsDocument).all()
            for owns_document in owns_documents:
                if (
                    owns_document.user_id == user_id
                    and owns_document.document_id == document.id
                ):
                    return True
    return False


@app.route("/login", methods=["POST"])
def login():
    data = request.json
    if not data or "username" not in data or "password" not in data:
        return jsonify({"message": "Missing username or password"}), 400

    username = data["username"]
    password = data["password"]

    user = authenticate(username, password)

    if user:
        token = jwt.encode(
            {
                "identity": user["username"],
                "exp": datetime.utcnow() + timedelta(days=60),
            },
            app.config["SECRET_KEY"],
            algorithm="HS256",
        )
        return jsonify({"access_token": token, "name": user["name"]}), 200
    else:
        return jsonify({"message": "Invalid username or password"}), 401


@app.route("/register", methods=["POST"])
def register():
    data = request.json
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")

    encrypted_passwored = sha256_crypt.encrypt(password)
    new_user = models.User(
        name=name, username=email, password=encrypted_passwored, type="lecturer"
    )
    db.session.add(new_user)
    db.session.commit()

    return "New user added"


@app.route("/dashboard", methods=["GET"])
def getDashboard():
    course_data = []

    courses = db.session.query(models.Course).all()

    for course in courses:
        course_entry = {"id": course.id, "course": course.name, "resources": []}

        for topic in course.children:
            for document in sorted(
                topic.children, key=lambda x: x.date_created, reverse=True
            ):
                popularity = (
                    db.session.query(func.count(models.OwnsDocument.user_id))
                    .join(
                        models.Document,
                        models.OwnsDocument.document_id == models.Document.id,
                    )
                    .filter(models.Document.content == document.content)
                    .distinct()
                    .scalar()
                )
                document_entry = {
                    "id": document.id,
                    "title": document.title,
                    "topic": document.topic.name,
                    "rating": document.ratings,
                    "popularity": popularity,
                    "link": document.link,
                    "comment": document.comment,
                }

                course_entry["resources"].append(document_entry)

        course_data.append(course_entry)

    return jsonify(course_data)


@app.route("/rating", methods=["PUT"])
def edit_rating():
    try:
        document_id = request.args.get("document_id")
        rating = request.args.get("rating")

        document = (
            db.session.query(models.Document)
            .filter(models.Document.id == document_id)
            .first()
        )

        if not document:
            return jsonify({"error": "Document not found"}), 404

        document.ratings = rating
        db.session.commit()

        return jsonify(
            {
                "message": f'Rating for document with ID {document_id} updated to "{rating}"'
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # 500 Internal Server Error


@app.route("/comment", methods=["POST"])
def add_comment():
    try:
        document_id = request.args.get("document_id")
        comment = request.args.get("comment")

        document = (
            db.session.query(models.Document)
            .filter(models.Document.id == document_id)
            .first()
        )

        if not document:
            return jsonify({"error": "Document not found"}), 404

        document.comment = comment
        document.comment_date_added = datetime.utcnow()
        db.session.commit()

        return jsonify(
            {"message": f"Comment added for document with ID {document_id}."}
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # 500 Internal Server Error


@app.route("/add_embedding_comment", methods=["POST"])
@token_required
def add_embedding_comment():
    try:
        document_id = request.args.get("document_id")
        embedding_id = request.args.get("embedding_id")
        comment = request.args.get("comment")

        embedding = (
            db.session.query(models.Embeddings)
            .filter(
                models.Embeddings.document_id == document_id,
                models.Embeddings.id == embedding_id,
            )
            .first()
        )

        if not embedding:
            return jsonify({"error": "Embedding not found"}), 404
        
        new_comment = models.Comments(
            comment=comment,
            comment_date_added=datetime.utcnow(),
        )

        embedding.children.append(new_comment)

        db.session.commit()

        return jsonify(
            {
                "message": f"Comment added for document with ID {document_id} and embedding with ID {embedding_id}."
            }
        )

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500  # 500 Internal Server Error


@app.route("/get_all_comments", methods=["GET"])
@token_required
def get_all_comments():
    try:
        all_comments = {
            "embeddingComment": [],
            "otherEmbeddingComments": [],
            "lecturerComment": [],
        }

        document_id = request.args.get("document_id")
        embedding_id = request.args.get("embedding_id")

        document = (
            db.session.query(models.Document)
            .filter(
                models.Document.id == document_id,
            )
            .first()
        )
        if not document:
            return jsonify({"error": "Document not found"}), 404

        if document.comment:
            document_comment = {
                "comment": document.comment,
                "timestamp": None,
                "is_lecturer_comment": True,
                "page_number": None,
                "comment_date_added": document.comment_date_added,
            }

            all_comments["lecturerComment"].append(document_comment)
            
        embedding = (
            db.session.query(models.Embeddings)
            .filter(
                models.Embeddings.document_id == document_id,
                models.Embeddings.id == embedding_id,
            )
            .first()
        )
        if not embedding:
            return jsonify({"error": "Embedding not found"}), 404

        embedding_comments = embedding.children  
        for comment in embedding_comments:
            page = None
            formatted_timestamp = None

            if embedding.timestamp:
                timestamp = timedelta(seconds=embedding.timestamp)
                hours, remainder = divmod(timestamp.seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                formatted_timestamp = "{:02}:{:02}:{:02}".format(
                    int(hours), int(minutes), int(seconds)
                )

            if embedding.page:
                page = embedding.page

            all_comments["embeddingComment"].append(
                {
                    "comment": comment.comment,
                    "timestamp": formatted_timestamp,
                    "is_lecturer_comment": False,
                    "page_number": page,
                    "comment_date_added": comment.comment_date_added,
                }
            )

        other_embeddings = (
            db.session.query(models.Embeddings)
            .filter(
                models.Embeddings.document_id == document_id,
                models.Embeddings.id != embedding_id,
            )
            .all()
        )

        for other_embedding in other_embeddings:
            other_embedding_comments = other_embedding.children
            if other_embedding_comments:
                for comment in other_embedding_comments:
                    page = None
                    formatted_timestamp = None

                    if other_embedding.timestamp:
                        timestamp = timedelta(seconds=other_embedding.timestamp)
                        hours, remainder = divmod(timestamp.seconds, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        formatted_timestamp = "{:02}:{:02}:{:02}".format(
                            int(hours), int(minutes), int(seconds)
                        )
                    if other_embedding.page:
                        page = other_embedding.page

                    all_comments["otherEmbeddingComments"].append(
                        {
                            "comment": comment.comment,
                            "timestamp": formatted_timestamp,
                            "is_lecturer_comment": False,
                            "page_number": page,
                            "comment_date_added": comment.comment_date_added,
                        }
                    )

        return jsonify({"comments": all_comments})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500  # 500 Internal Server Error


@app.route("/course", methods=["POST"])
def handle_course():
    try:
        data = request.json
        courses = data.get("courses")

        # course_name = data.get('courseName')
        # course_code = data.get('courseCode')
        # topics = data.get('topics', [])

        # new_course = models.Course(name=course_name, code=course_code)

        # for topic_data in topics:
        #     topic_name = topic_data.get('topicName')
        #     subtopics = topic_data.get('subtopics', [])

        #     # Create a new Topic instance
        #     new_topic = models.Topic(name=topic_name)
        #     new_course.topics.append(new_topic)

        #     # # Add subtopics to the new topic
        #     # for subtopic_data in subtopics:
        #     #     subtopic_name = subtopic_data.get('subtopicName')
        #     #     new_subtopic = Subtopic(name=subtopic_name)
        #     #     new_topic.subtopics.append(new_subtopic)

        # db.session.add(new_course)
        # db.session.commit()

        for course_data in courses:
            course = models.Course(
                name=course_data["courseName"], code=course_data["courseCode"]
            )
            db.session.add(course)

            for topic_data in course_data["topics"]:
                topic = models.Topic(name=topic_data["topicName"])
                course.children.append(topic)

                db.session.add(topic)

                topic_embedding = embeddings_model.embed_query(topic.name)
                topic.embedding = topic_embedding

        db.session.commit()
        return jsonify(message="Course details added successfully")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get-all-courses", methods=["GET"])
def filter_type():
    courses = db.session.query(models.Course.name).all()

    if not courses:
        return jsonify({"error": "Topic not found"}), 404

    course_list = [course.name for course in courses]

    return jsonify({"topics": course_list}), 200


@app.route("/update-label", methods=["PUT"])
def update_label():
    try:
        document_id = request.args.get("document_id")
        label = request.args.get("label")

        document = (
            db.session.query(models.Document)
            .filter(models.Document.id == document_id)
            .first()
        )

        if not document:
            return jsonify({"error": "Document not found"}), 404

        document.label = label
        db.session.commit()

        return jsonify(
            {
                "message": f'Label for document with ID {document_id} updated to "{label}"'
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500  # 500 Internal Server Error
