DB_USERNAME = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "yt_vec"
DB_PASSWORD = "your_password"

SQLALCHEMY_DATABASE_URI = f"postgresql+psycopg2://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
#SQLALCHEMY_DATABASE_URI = "postgresql+psycopg2://postgres:root@localhost:5432/yt_vec"
SQLALCHEMY_TRACK_MODIFICATIONS = False
APP_PORT = 8080
