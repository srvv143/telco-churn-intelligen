import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(bundle_path):
    import api.main as main

    original = main.MODEL_PATH
    main.MODEL_PATH = bundle_path
    with TestClient(main.app) as c:
        yield c
    main.MODEL_PATH = original


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_model_info(client):
    body = client.get("/v1/model").json()
    assert 0.5 < body["test_roc_auc"] <= 1
    assert body["actions"] and "gender" in body["features_excluded"]


def test_predict_single(client, customer):
    r = client.post("/v1/predict", json=customer)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["customerID"] == customer["customerID"]
    assert body["risk_drivers"], "reason codes expected by default"
    assert "X-Process-Time-Ms" in r.headers


def test_predict_without_explanation(client, customer):
    body = client.post("/v1/predict?explain=false", json=customer).json()
    assert body["risk_drivers"] == []


def test_invalid_category_rejected(client, customer):
    customer["Contract"] = "Weekly"
    assert client.post("/v1/predict", json=customer).status_code == 422


def test_inconsistent_services_rejected(client, customer):
    customer["InternetService"] = "No"  # add-ons still say "No"/"Yes"
    r = client.post("/v1/predict", json=customer)
    assert r.status_code == 422
    assert "No internet service" in r.text


def test_negative_tenure_rejected(client, customer):
    customer["tenure"] = -1
    assert client.post("/v1/predict", json=customer).status_code == 422


def test_batch(client, customer):
    second = {
        **customer,
        "customerID": "DEMO-0002",
        "Contract": "Two year",
        "tenure": 60,
        "TotalCharges": 5900.0,
    }
    r = client.post("/v1/predict/batch", json={"customers": [customer, second]})
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["n_customers"] == 2
    probs = {p["customerID"]: p["churn_probability"] for p in body["predictions"]}
    assert probs["DEMO-0001"] > probs["DEMO-0002"]


def test_empty_batch_rejected(client):
    assert client.post("/v1/predict/batch", json={"customers": []}).status_code == 422
