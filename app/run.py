from app import app
from app.config import APP_PORT
import os

if __name__ == "__main__":
    #app.secret_key = os.urandom(12)
    app.secret_key = "secret_key"
    app.run(debug=True, port=APP_PORT)
