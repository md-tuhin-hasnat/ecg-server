from main import app  # Import your FastAPI app
from fastapi.middleware.wsgi import WSGIMiddleware

# Wrap it as WSGI app for compatibility
application = WSGIMiddleware(app)
