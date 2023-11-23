from flask import Blueprint, request, current_app

from langchain.embeddings import HuggingFaceEmbeddings

from app import models
from models import db

from llama_hub.youtube_transcript import YoutubeTranscriptReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index import download_loader
import whisper
from utils import (
    get_keywords,
    preprocess_text,
    assign_topic,
    calculate_and_store_similarity,
    allowed_file,
)
from werkzeug.utils import secure_filename
import os
import threading
from pathlib import Path
from pytube import YouTube

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

embed_routes = Blueprint("embed_routes", __name__)


@embed_routes.route("/embed_youtube", methods=["POST"])
def embed_youtube():
    loader = YoutubeTranscriptReader()
    data = request.json

    video_url = data.get("video_url")
    yt = YouTube(video_url)
    title = yt.title

    documents = loader.load_data(ytlinks=[video_url])
    transcript_text = documents[0].text
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

    return "Embeddings saved in the database."


@embed_routes.route("/embed_pdf", methods=["POST"])
def embed_pdf():
    loader = PDFReader()
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
        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
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

        return "Embeddings saved in the database."


@embed_routes.route("/embed_pptx", methods=["POST"])
def embed_pptx():
    loader = PptxReader()
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
        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
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

        return "Embeddings saved in the database."


@embed_routes.route("/embed_audio", methods=["POST"])
def embed_audio():
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
        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
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

        return "Audio embeddings saved in the database."
    else:
        return "Invalid audio file format. Allowed extensions: mp3, mp4, mpeg, mpga, m4a, wav, webm"


@embed_routes.route("/embed_docx", methods=["POST"])
def embed_docx():
    loader = DocxReader()
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
        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
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

        return "Embeddings saved in the database."
