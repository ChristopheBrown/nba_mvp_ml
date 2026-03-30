from __future__ import annotations

from flask import Flask

from flask_app.config import Config
from flask_app.control import control_bp
from flask_app.models import ModelHandler
from flask_app.routes import api_blueprint, configure_routes


def create_app(model_handler: ModelHandler | None = None) -> Flask:
    """Application factory used by Flask and gunicorn."""

    app = Flask(__name__)
    app.config.from_object(Config)

    handler = model_handler or ModelHandler()
    configure_routes(handler)

    if model_handler is None:
        handler.load_model()

    app.register_blueprint(api_blueprint)
    app.register_blueprint(control_bp)
    return app
