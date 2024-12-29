# NBA MVP Prediction Project

## Overview

This project is a full-stack machine learning solution designed to predict the NBA Most Valuable Player (MVP) award with a focus on creating an end-to-end deployable system. By leveraging data pipelines, machine learning models, and API integrations, this project offers insights into the MVP race and showcases engineering expertise.

The system is tailored to be a portfolio piece, highlighting skills in data engineering, machine learning, and deployment pipelines. It is particularly aimed at hiring committees for machine learning engineering roles.

---

## Key Objectives

1. **Predict NBA MVP**: Build an accurate, robust, and deployable system for predicting the MVP.
2. **End-to-End Solution**: Integrate data pipelines, machine learning models, and APIs into a cohesive architecture.
3. **Portfolio Value**: Showcase the ability to design, develop, and deploy complex ML systems.
4. **Real-Time Potential**: Lay the groundwork for real-time predictions and data ingestion.

---

## Core System Architecture

1. **Data Ingestion and Preprocessing**: 
   - **Tool**: Mage.
   - **Sources**: Historical MVP voting, team stats, and player stats from Basketball-Reference.
   - **Output**: A cleaned dataset merging relevant statistics for MVP finalists.

2. **Model Training and Experimentation**:
   - **Tool**: MLFlow.
   - **Models**: Neural networks, Random Forest, and XGBoost. Neural networks were chosen for their ability to generalize with high-dimensional input.
   - **Registry**: Models are logged and versioned in MLFlow for easy deployment.

3. **API for Predictions**:
   - **Tool**: Flask.
   - **Functionality**: Expose RESTful endpoints to make predictions using the MLFlow-deployed model.

4. **Deployment**:
   - **Local Deployment**: Flask application and MLFlow model running locally.
   - **Future Plan**: Transition to AWS for a cloud-based deployment, with containerization via Docker.

---

## Key Achievements

- **End-to-End Data Pipeline**: Successfully implemented a Mage pipeline to handle historical data and transform it for model training.
- **Model Deployment**: MLFlow serves the trained model, with Flask acting as an intermediary to handle prediction requests.
- **API Integration**: Developed a Flask API capable of accepting inputs and returning predictions from the deployed ML model.
- **Testing**: Verified functionality of API and prediction endpoints using both Python scripts and `curl` commands.
- **Project Structure**: Organized project files to separate concerns effectively, from data ingestion to model deployment.

---

## Example Usage

### Running the Flask API
```bash
export FLASK_APP=flask_app
flask run --host 0.0.0.0 --port 5002
```

### Sending a Prediction Request
Using Python
```python
import requests
import numpy as np

url = "http://127.0.0.1:5002/predict"
data = np.random.rand(1, 24).tolist()  # Replace with real input data
response = requests.post(url, json=data)
print(response.json())
```

Using curl
```bash
curl -X POST http://127.0.0.1:5002/predict \
-H "Content-Type: application/json" \
-d '[[-0.179, 0.214, ... ]]'
```

## Challenges Encountered
1. **Data Complexity**: 
   - Handling edge cases like Dominique Wilkins, who played for two teams in a single season.
   - Ensuring alignment of team and player stats across multiple sources.

2. **Technical Roadblocks**:
   - Resolving issues with Flask’s connection to MLFlow.
   - Debugging input schema mismatches for MLFlow-deployed models.
   - Networking complexities between Flask and MLFlow services.

3. **Compute Costs**:
   - Optimizing the sentiment analysis to minimize API usage costs for GPT-4.
  
## Project Structure

The project is organized into a well-structured hierarchy to ensure modularity and maintainability. Below is an overview of the key directories and their purposes:

```plaintext
nba_mvp_ml/
├── flask_app/               # Contains the Flask application for serving predictions
│   ├── __init__.py          # Initializes the Flask app and sets up configurations
│   ├── routes.py            # Defines API routes, including the `/predict` endpoint
│   ├── models.py            # Handles model loading and interactions with MLFlow
│   ├── utils.py             # Utility functions used across the Flask app
│   ├── templates/           # (Optional) Holds HTML templates for any front-end components
│   ├── static/              # (Optional) Stores static files like CSS/JS
├── notebooks/               # Jupyter notebooks for data exploration and model experimentation
│   ├── data_analysis.ipynb  # Notebook for analyzing raw data and generating insights
│   ├── model_training.ipynb # Notebook for training and logging models in MLFlow
├── pipelines/               # Mage pipeline definitions for data ingestion and processing
│   ├── nba_data_scraping/   # Pipeline for scraping and preprocessing NBA data
│   ├── transformations/     # Custom data transformations for feature engineering
│   ├── data_loaders/        # Scripts for loading data from various sources
├── mlruns/                  # MLFlow tracking directory for experiment metadata and artifacts
├── data/                    # Contains raw and processed data files
│   ├── raw/                 # Raw data downloaded from external sources
│   ├── processed/           # Processed and feature-engineered datasets
├── models/                  # Serialized machine learning models (if stored locally)
│   ├── serialized_model.pt  # Example serialized PyTorch model
├── Dockerfile               # Dockerfile for containerizing the Flask app and dependencies
├── requirements.txt         # Python dependencies for the project
├── README.md                # Comprehensive project documentation
├── metadata.yaml            # Metadata for Mage pipelines
├── mlflow.db                # SQLite database for MLFlow tracking (local use)
├── .gitignore               # Specifies files and directories to be ignored by Git
```

## Key Learnings

- **System Integration**: Learned how to create seamless communication between data pipelines, machine learning models, and API layers, ensuring a cohesive workflow from data ingestion to prediction delivery.
- **MLFlow Expertise**: Gained hands-on experience with MLFlow’s model registry, versioning, and deployment workflows. Addressed challenges in serving models and understanding input schema enforcement.
- **Error Resolution**: Developed skills in debugging complex issues such as input schema mismatches, Flask-MLFlow integration, and local server conflicts.
- **API Design and Testing**: Designed and tested a robust Flask API to serve predictions from a deployed model, understanding the importance of adhering to strict data schemas for production-ready systems.
- **Sentiment Analysis Optimization**: Improved the sentiment scoring process to reduce API usage costs while maintaining valuable insights for the MVP prediction model.

---

## Limitations and Future Goals

### Limitations

1. **Narrow Dataset**: The dataset focuses on MVP finalists and does not generalize to other use cases or awards, limiting the scope of predictions.
2. **Subjectivity in Sentiment Analysis**: The reliance on GPT-based sentiment scoring introduces subjectivity, which might not align perfectly with real-world voting patterns.
3. **Local Deployment**: The current deployment is limited to local systems, which restricts broader access and scalability.
4. **Static Data**: Predictions are based on historical data without incorporating live updates, which limits its real-time utility.

### Future Goals

1. **Containerization**: Dockerize the Flask API and MLFlow services to enhance portability and simplify deployment in various environments.
2. **Cloud Deployment**: Migrate the system to AWS (using EC2 or Lambda) to enable scalability, reliability, and global accessibility.
3. **Real-Time Updates**: Implement live data ingestion to provide real-time MVP predictions during the NBA season.
4. **Feature Expansion**: Broaden the dataset to include all players, enabling predictions for other awards such as Defensive Player of the Year or Rookie of the Year.
5. **Visualization Dashboard**: Develop an interactive dashboard for displaying predictions, insights, and comparisons to enhance user engagement and interpretability.
6. **Advanced Sentiment Analysis**: Refine sentiment scoring methods to reduce subjectivity and align closer with actual voting behaviors.

---

## Conclusion

This project demonstrates the creation of an end-to-end machine learning pipeline for predicting the NBA MVP award, highlighting expertise in data engineering, machine learning, and API integration. By combining tools like Mage for data pipelines, MLFlow for model management, and Flask for API deployment, it showcases a seamless approach to solving real-world problems. 

The system is designed with scalability and extensibility in mind, offering clear pathways for future enhancements such as containerization, cloud deployment, and real-time predictions. 
