.PHONY: install clean lint test demo docker-demo

install:
	pip install -r requirements.txt
	touch .venv-placeholder  # reminder to use your preferred env

clean:
	rm -rf .pytest_cache */__pycache__

lint:
	python -m compileall flask_app tests

test:
	python -m pytest -q

demo:
	FLASK_RUN_PORT=5000 MVP_MODEL_ARTIFACT_PATH=$$(pwd)/mlops/artifacts/24-nn-1 \
		python -m flask --app flask_app run --host 0.0.0.0 --port 5000

docker-demo:
	docker-compose up --build
