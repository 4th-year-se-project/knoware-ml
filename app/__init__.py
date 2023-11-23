import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from app.config import SQLALCHEMY_DATABASE_URI
from flask_migrate import Migrate
from flask_cors import CORS

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = SQLALCHEMY_DATABASE_URI
db = SQLAlchemy(app)
migrate = Migrate(app, db)

CORS(app)

app.config["UPLOAD_FOLDER"] = "uploads"
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

from app.routes.search_routes import search_routes
from app.routes.embed_routes import embed_routes
from app.routes.recommend_routes import recommend_routes
from app.routes.resource_routes import resource_routes

app.register_blueprint(search_routes)
app.register_blueprint(embed_routes)
app.register_blueprint(recommend_routes)
app.register_blueprint(resource_routes)

from app import models
