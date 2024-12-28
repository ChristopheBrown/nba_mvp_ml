from flask import Flask
from flask_app.routes import api_blueprint
from flask_app.config import Config

def create_app():
    """App Factory for Flask application."""
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(Config)
    
    # Register API routes
    app.register_blueprint(api_blueprint)
    
    return app