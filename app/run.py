from app import app
from app.config import APP_PORT
import os

if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True, host='0.0.0.0', port=APP_PORT)

