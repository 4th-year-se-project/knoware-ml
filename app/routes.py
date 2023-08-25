import os
from app import app, db, models
from flask import request
from llama_hub.youtube_transcript import YoutubeTranscriptReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from werkzeug.utils import secure_filename
from llama_index import download_loader
from pathlib import Path
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from IPython.display import display
import whisper


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


# Configure the upload folder
app.config["UPLOAD_FOLDER"] = "uploads"
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

ALLOWED_EXTENSIONS = {"mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/embed_youtube", methods=["POST"])
def embed_youtube():
    loader = YoutubeTranscriptReader()
    print("started")
    data = request.json
    video_url = data.get("video_url")
    documents = loader.load_data(ytlinks=[video_url])
    transcript_text = documents[0].text
    print("loaded")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_text(transcript_text)
    embeddings = embeddings_model.embed_documents(docs)
    print("embedded")
    for embedding, doc in zip(embeddings, docs):
        stored_embedding = models.Document(
            content=doc, embedding=embedding, title=video_url
        )
        db.session.add(stored_embedding)
    db.session.commit()

    return "Embeddings saved in the database."


@app.route("/embed_pdf", methods=["POST"])
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
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Load and process the PDF content
        documents = loader.load_data(
            file=Path(filepath)
        )  # Implement the PDF loading function
        print(documents)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150
        )
        docs = text_splitter.split_text(documents[0].text)

        embeddings = embeddings_model.embed_documents(docs)

        # Store the embeddings in the database
        for embedding, doc in zip(embeddings, docs):
            stored_embedding = models.Document(
                content=doc, embedding=embedding, title=filename
            )
            db.session.add(stored_embedding)
        db.session.commit()

        return "Embeddings saved in the database."


@app.route("/embed_pptx", methods=["POST"])
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
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Load and process the PDF content
        documents = loader.load_data(
            file=Path(filepath)
        )  # Implement the PDF loading function
        print(documents)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150
        )
        docs = text_splitter.split_text(documents[0].text)

        embeddings = embeddings_model.embed_documents(docs)

        # Store the embeddings in the database
        for embedding, chunk in zip(embeddings, docs):
            stored_embedding = models.Document(
                content=chunk, embedding=embedding, title=filename
            )
            db.session.add(stored_embedding)
        db.session.commit()

        return "Embeddings saved in the database."


@app.route("/embed_audio", methods=["POST"])
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
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        audio_file.save(filepath)

        # Load and process the audio content
        transcript = whisper_model.transcribe(filepath)
        print(transcript)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150
        )
        docs = text_splitter.split_text(transcript["text"])

        embeddings = embeddings_model.embed_documents(docs)

        # Store the embeddings in the database
        for embedding, chunk in zip(embeddings, docs):
            stored_embedding = models.Document(
                content=chunk, embedding=embedding, title=filename
            )
            db.session.add(stored_embedding)
        db.session.commit()

        return "Audio embeddings saved in the database."
    else:
        return "Invalid audio file format. Allowed extensions: mp3, mp4, mpeg, mpga, m4a, wav, webm"


@app.route("/embed_docx", methods=["POST"])
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
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Load and process the docx content
        documents = loader.load_data(
            file=Path(filepath)
        )  # Implement the docx loading function
        print(documents)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150
        )
        docs = text_splitter.split_text(documents[0].text)

        embeddings = embeddings_model.embed_documents(docs)

        # Store the embeddings in the database
        for embedding, doc in zip(embeddings, docs):
            stored_embedding = models.Document(
                content=doc, embedding=embedding, title=filename
            )
            db.session.add(stored_embedding)
        db.session.commit()

        return "Embeddings saved in the database."


@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query")
    query_embedding = embeddings_model.embed_query(query)
    results = db.session.query(models.Document)
    results = results.order_by(
        models.Document.embedding.cosine_distance(query_embedding)
    ).limit(5)
    return {"results": [result.content for result in results]}
