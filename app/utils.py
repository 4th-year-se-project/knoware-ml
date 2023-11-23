from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from app import app, models
from models import db

modelPath = "../models/all-MiniLM-L6-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
embeddings_model = HuggingFaceEmbeddings(
    model_name=modelPath, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

sentence_model = SentenceTransformer(modelPath)
kw_model = KeyBERT(model=sentence_model)

ALLOWED_EXTENSIONS = {"mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"}


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


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
