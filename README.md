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
5. **Feature Contract Discipline**: Maintain an explicit schema (see `json/feature_schema_v1.json`) and runtime builder (`src/features/feature_builder.py`) so scoring vectors always match the locked v1 order and normalization expectations.

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

## Runtime Feature Builder
- **Candidate pool ranking**: `scripts/build_candidate_pool_vectors.py` builds the normalized vectors, saves the latest scaler params (`json/scaler_params_v1.json`), and exports the top `N` candidate probabilities + feature vectors under `data_exporters/candidate_pool/`. Use the script (or a cron job) to refresh the ranking artifacts before pushing API updates.


- **Schema contract**: `json/feature_schema_v1.json` locks the ordered 24-feature vector for the MVP model and documents which stats, advanced metrics, and sentiment signals are expected in each position.
- **Runtime enforcement**: `src/features/feature_builder.py` consumes the schema artifact (plus optional `mean`/`scale` arrays) to validate inputs, normalize using stored scaler parameters, and emit `FeatureVector` records that carry metadata for inspection before scoring.
- **Demo script**: `scripts/build_runtime_feature_vectors.py` accepts player feature mappings (examples live in `json/sample_player_features.json`) and writes normalized vectors to disk so you can verify schema adherence without ingesting the entire data lake.
- **Runtime pipeline**: `src/features/pipeline.py` loads the season totals, derives TAP-level stats (PER, BPM, Win Shares, TOV%, ORtg, etc.), and merges the sample sentiment bundle at `json/sample_sentiment_scores.json` so runtime vectors stay aligned with the locked schema.
- **Candidate builder**: `scripts/build_candidate_pool_vectors.py --season 2023 --top-n 30` (or adjust the season) materializes both `output/candidate_feature_vectors.json` and the latest `json/scaler_params_v1.json`, giving the runtime feature builder the normalization parameters it needs before scoring.



## Candidate Pool Rankings

- **CLI export**: `scripts/build_candidate_pool_vectors.py --season <year> --pool-size 30 --top-n 5` fits the training scaler, normalizes the pool, scores the MVP model, and writes both the ranking payload (`data_exporters/candidate_pool/latest_candidate_pool.json`) and the schema-aligned feature vectors (`data_exporters/candidate_pool/candidate_feature_vectors.json`). It also refreshes `json/scaler_params_v1.json` so the runtime builder stays synchronized.

- **HTTP endpoint**: `GET /candidate_pool` (query params: `top_n`, `pool_size`, `season`, `mode`, `cursor`). The response includes `feature_schema_version`, `scaler_version`, `model_version`, deterministic metadata (`pool_size`, `next_cursor`, `snapshot_timestamp`), and the ranked `results` array (each entry includes `player_id`, `player_name`, `mvp_probability`, `not_mvp_probability`, `mvp_rank`, and the builder metadata). Supply `cursor` to page through the sorted pool in a keyset-friendly way.

## Testing and Validation

- **Contract checks**: `tests/test_feature_builder.py` proves the runtime builder raises helpful errors when features are missing, preserves metadata through batch builds, and applies normalization when scaler params are provided.

## API Contracts & Monitoring

- **/predict**: Accepts the 24-float schema and returns `predictions`, `count`, and `model_version` so clients know exactly which artifact generated the score.
- **/candidate_pool**: Returns `feature_schema_version`, `scaler_version`, `model_version`, pagination tokens (`cursor`/`next_cursor`), and the deterministic ranked `results` array. Clients can re-run the CLI export or hit this endpoint with `cursor` to stream the top-N rankings consistently.
- **Monitoring hooks**: `src/monitoring.py` emits structured metric logs that drive dashboards or Prometheus-style exporters. Current metric names include `stats_rows_loaded`, `sentiment_entries_processed`, `sentiment_missing_keys`, and `candidate_pool_scored`, covering the stats/sentiment ingestion and the candidate-pool scoring runs.

## Example Usage

### Demo-ready serving (local or Docker)

The packaged `24-nn-1` MLflow artifact lives under `mlops/artifacts/24-nn-1`. The quickest entry points are:

#### 1. Prepare a Python 3.12 sandbox
- Install Python 3.12 (for example, `brew install python@3.12` on macOS or use your distro’s `python3.12` package, `pyenv install 3.12.13`, etc.).
- Create a fresh virtual environment: `python3.12 -m venv .venv312`.
- Activate it: `source .venv312/bin/activate`.
- Install the dependencies: `pip install -r requirements.txt`.
- Confirm that `mlops/artifacts/24-nn-1` exists (or point `MVP_MODEL_ARTIFACT_PATH` at your packaged artifact).

Running the demo with any Python interpreter older than 3.12 (the default macOS `/usr/bin/python3` is 3.9.6) currently raises `TypeError: code() takes at most 16 arguments (18 given)` inside `torch.load`. Using a 3.12 interpreter resolves that compatibility issue because the saved code objects match the artifact.

#### 2. Launch the Flask service
With the sandbox active, run `make demo`. The Makefile already exports `FLASK_RUN_PORT` and `MVP_MODEL_ARTIFACT_PATH` so the packaged artifact is picked up automatically. If you prefer to run Flask manually, use:
```bash
FLASK_RUN_PORT=5000 MVP_MODEL_ARTIFACT_PATH=$(pwd)/mlops/artifacts/24-nn-1 \
python -m flask --app flask_app run --host 0.0.0.0 --port 5000
```
Once the artifact loads you will see the Werkzeug banner announcing `Running on http://127.0.0.1:5000`.

#### 3. Hit `/predict` for a sanity check
```bash
curl -s -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}'
```
Sample response from the zero-vector input:
```json
{
  "count": 1,
  "predictions": [[0.999840497970581, 0.00015953574620652944]]
}
```

#### 4. Docker / gunicorn
`make docker-demo` builds the image from `Dockerfile`, wires gunicorn, and exposes port `8000` via `docker-compose`. The compose file already wires `MVP_MODEL_ARTIFACT_PATH` so the packaged artifact is mounted by default.

Full, step-by-step instructions (env vars, smoke tests, curl examples) now live in `docs/demo.md`.

### Running the Flask API (legacy instructions)
If you still want the original bare `flask run` flow, it has been retained in `docs/demo.md` under “Legacy manual run.” The same Python 3.12 sandbox works for those steps as well.

### Prediction request schema
The `/predict` endpoint expects a JSON body with a `features` array of exactly 24 floats (see `flask_app/schemas.py`). The sample call above proves the format; swap in real feature vectors as needed to reproduce MVP scores.

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
