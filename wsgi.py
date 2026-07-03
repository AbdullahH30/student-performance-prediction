"""
wsgi.py – WSGI entry-point for production servers (Gunicorn, uWSGI).

Usage:
    gunicorn --bind 0.0.0.0:5000 wsgi:application
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.app import create_app

application = create_app()

if __name__ == "__main__":
    application.run(debug=False, host="0.0.0.0", port=5000)
