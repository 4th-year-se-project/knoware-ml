import os
from app import app, db, models
from flask import request
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


@app.route("/embed_youtube", methods=["POST"])
def embed_youtube():
    loader = YoutubeTranscriptReader()
    data = request.json
    video_url = data.get("video_url")
    documents = loader.load_data(ytlinks=[video_url])
    transcript_text = documents[0].text
    preprocessed_transcript_text = preprocess_text(transcript_text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_text(transcript_text)
    embeddings = embeddings_model.embed_documents(docs)
    keywords = get_keywords(docs)
    stored_document = models.Document(
        title=video_url, content=preprocessed_transcript_text, keywords=keywords
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


@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query")
    print(query)
    query_embedding = embeddings_model.embed_query(query)

    # Create a dictionary to store the results, indexed by document ID
    results_dict = {}

    # Perform a join between Embeddings and Document tables
    results = db.session.query(models.Embeddings, models.Document)
    results = results.join(
        models.Document, models.Embeddings.document_id == models.Document.id
    )

    # Calculate and order by cosine distance
    results = results.order_by(
        models.Embeddings.embedding.cosine_distance(query_embedding)
    )

    for result in results:
        embedding, document = result
        doc_id = document.id
        # Check if this document ID is already in the results_dict
        if doc_id not in results_dict:
            results_dict[doc_id] = {
                "title": document.title,
                "content": embedding.split_content,
                "doc_id": doc_id,
            }

    # Convert the results_dict values to a list
    response_data = list(results_dict.values())

    return {"results": response_data}


def assign_topic(stored_document):
    embeddings = (
        db.session.query(models.Embeddings)
        .filter(models.Embeddings.document_id == stored_document.id)
        .all()
    )
    topics = db.session.query(models.Topic).all()

    best_topic = None
    best_subtopic = None
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

    best_similarity = -1

    subtopics = (
        db.session.query(models.SubTopic)
        .filter(models.SubTopic.topic_id == best_topic)
        .all()
    )

    # Iterate through document chunks and calculate similarities with the subtopics
    for embedding in embeddings:
        for subtopic in subtopics:
            similarity = cosine_similarity([embedding.embedding], [subtopic.embedding])[
                0
            ][0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_subtopic = subtopic.id

    # Assign best matching subtopic to the document
    if best_subtopic is not None:
        stored_document.subtopic_id = best_subtopic
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

    return list(keyword_set)
