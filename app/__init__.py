from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from app.config import SQLALCHEMY_DATABASE_URI
from flask_migrate import Migrate
from flask_cors import CORS 

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = SQLALCHEMY_DATABASE_URI
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
db = SQLAlchemy(app)
migrate = Migrate(app, db)

#CORS(app, resources={r"/api/*": {"origins": "https://knoware-frontend.vercel.app/"}})
#CORS(app)

#CORS(
#    app,
#    resources={r"/api/*": {"origins": "https://knoware-frontend.vercel.app"}},
#    supports_credentials=True,
#    allow_headers=["Content-Type", "Authorization"],
#    expose_headers=["Content-Type", "Authorization"],
#)

from app import routes, models
