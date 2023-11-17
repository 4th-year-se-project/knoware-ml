from app import app
from config import APP_PORT

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=APP_PORT)
