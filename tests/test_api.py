from flask_app import create_app

EXPECTED_FEATURES = 24


class FakeModelHandler:
    model_version = "fake-test-model"

    def load_model(self):
        return self

    def predict(self, _input):
        return [123.45]


def _make_client():
    app = create_app(model_handler=FakeModelHandler())
    app.testing = True
    return app.test_client()


def test_predict_endpoint_returns_predictions():
    client = _make_client()
    payload = {"features": [0.0] * EXPECTED_FEATURES}

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.get_json()
    assert data["predictions"] == [123.45]
    assert data["count"] == 1
    assert data["model_version"] == "fake-test-model"


def test_predict_endpoint_validates_payload():
    client = _make_client()
    payload = {"features": [0.0] * 10}  # too short

    response = client.post("/predict", json=payload)

    assert response.status_code == 400
    data = response.get_json()
    assert data["error"] == "Invalid payload."
