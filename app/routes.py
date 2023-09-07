from app import app, db, models
from flask import request
from llama_hub.youtube_transcript import YoutubeTranscriptReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
import threading

modelPath = "../models/all-MiniLM-L6-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
embeddings_model = HuggingFaceEmbeddings(
    model_name=modelPath, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
loader = YoutubeTranscriptReader()


@app.route("/embed_youtube", methods=["POST"])
def embed_youtube():
    data = request.json
    video_url = data.get("video_url")
    documents = loader.load_data(ytlinks=[video_url])
    transcript_text = documents[0].text
    preprocessed_transcript_text = preprocess_text(transcript_text)
    stored_document = models.Document(
        title=video_url, content=preprocessed_transcript_text
    )
    db.session.add(stored_document)
    db.session.commit()

    # Start a new thread to execute calculate_and_store_similarity
    similarity_thread = threading.Thread(
        target=calculate_and_store_similarity,
        args=(stored_document.id, preprocessed_transcript_text),
    )
    similarity_thread.start()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_text(transcript_text)
    embeddings = embeddings_model.embed_documents(docs)
    for embedding, doc in zip(embeddings, docs):
        stored_embedding = models.Embeddings(
            split_content=doc, embedding=embedding, document_id=stored_document.id
        )
        db.session.add(stored_embedding)
    db.session.commit()

    return "Embeddings saved in the database."


@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query")
    query_embedding = embeddings_model.embed_query(query)
    results = db.session.query(models.Embeddings)
    results = results.order_by(
        models.Embeddings.embedding.cosine_distance(query_embedding)
    ).limit(5)
    return {"results": [result.split_content for result in results]}


def calculate_and_store_similarity(new_document_id, new_document_text):
    with app.app_context():
        existing_documents = (
            db.session.query(models.Document)
            .filter(models.Document.id != new_document_id)
            .all()
        )

        if existing_documents:
            existing_document_texts = [doc.content for doc in existing_documents]

            tfidf_vectorizer = TfidfVectorizer()

            tfidf_matrix = tfidf_vectorizer.fit_transform(existing_document_texts)
            new_document_vector = tfidf_vectorizer.transform([new_document_text])
            similarity_scores = cosine_similarity(new_document_vector, tfidf_matrix)

            for existing_doc, similarity_score in zip(
                existing_documents, similarity_scores[0]
            ):
                doc_similarity = models.DocSimilarity(
                    new_document_id=new_document_id,
                    existing_document_id=existing_doc.id,
                    similarity_score=similarity_score,
                )
                db.session.add(doc_similarity)
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
