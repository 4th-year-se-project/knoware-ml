from app import app, db, models
from flask import request
from llama_hub.youtube_transcript import YoutubeTranscriptReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

modelPath = "../models/all-MiniLM-L6-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
embeddings_model = HuggingFaceEmbeddings(
    model_name=modelPath, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
loader = YoutubeTranscriptReader()


@app.route("/embed_youtube", methods=["POST"])
def embed_youtube():
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
