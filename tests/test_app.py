from pathlib import Path
import json
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as house_app
from model_io import ModelLoadError, load_trained_model


def test_model_info_endpoint_returns_metadata():
    client = house_app.app.test_client()

    response = client.get('/model_info')

    assert response.status_code == 200
    payload = response.get_json()
    assert 'metrics' in payload
    assert 'feature_importances' in payload


def test_predict_endpoint_rejects_invalid_json():
    client = house_app.app.test_client()

    response = client.post('/predict', data='not-json', content_type='application/json')

    assert response.status_code == 400
    assert response.get_json()['error'] == 'Request body must be valid JSON.'


def test_predict_endpoint_returns_price_and_breakdown():
    client = house_app.app.test_client()

    response = client.post(
        '/predict',
        json={
            'sqft_living': 2200,
            'bedrooms': 3,
            'bathrooms': 2.5,
            'age_years': 8,
            'distance_to_center_km': 4.2,
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert 'price' in payload
    assert 'breakdown' in payload
    assert payload['price'] > 0


def test_loader_rejects_model_trained_with_other_version(tmp_path):
    model_path = tmp_path / 'model.pkl'
    model_path.write_bytes(b'placeholder')
    metadata_path = tmp_path / 'model_metadata.json'
    metadata_path.write_text(
        json.dumps({'training_environment': {'scikit-learn': '0.0'}}),
        encoding='utf-8',
    )

    with pytest.raises(ModelLoadError, match='dependency mismatch'):
        load_trained_model(model_path, metadata_path)


def test_predict_explains_unavailable_model(monkeypatch):
    monkeypatch.setattr(house_app, 'model', None)

    def incompatible_model(*_args):
        raise ModelLoadError('Run `python train.py` again.')

    monkeypatch.setattr(house_app, 'load_trained_model', incompatible_model)
    response = house_app.app.test_client().post('/predict', json={'sqft_living': 1800})

    assert response.status_code == 503
    assert 'train.py' in response.get_json()['error']
