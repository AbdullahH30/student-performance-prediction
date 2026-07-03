"""
app.py – Flask application factory and entry-point.
"""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask


def create_app() -> Flask:
    """Create and configure the Flask application.

    Returns:
        Configured :class:`flask.Flask` instance.
    """
    template_dir = Path(__file__).parent / "templates"
    static_dir = Path(__file__).parent / "static"

    app = Flask(
        __name__,
        template_folder=str(template_dir),
        static_folder=str(static_dir),
    )
    app.secret_key = "spp-secret-2024-xyzabc"

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # ── Custom Jinja2 filters & globals ──────────────────────────────────
    app.jinja_env.globals["enumerate"] = enumerate
    app.jinja_env.globals["zip"] = zip

    def do_enumerate(iterable, start=0):
        return enumerate(iterable, start)

    app.jinja_env.filters["enumerate"] = do_enumerate

    # Inject `request` into every template context automatically
    from flask import request as _req

    @app.context_processor
    def inject_request():
        return {"request": _req}

    # Register all routes
    try:
        from app.routes import register_routes
    except ModuleNotFoundError:
        from routes import register_routes
    register_routes(app)

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(debug=True, host="0.0.0.0", port=5000)
