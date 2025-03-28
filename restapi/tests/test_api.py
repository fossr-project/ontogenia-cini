import io
import pytest
from fastapi.testclient import TestClient
from app.main import app
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
client = TestClient(app)
print([route.path for route in app.routes])
print("Routes loaded:")
for route in app.routes:
    print(route.path)

@pytest.mark.parametrize("use_default_dataset, file_upload", [
    (True, None),  # Test default dataset mode
    (False, io.BytesIO(b'gold standard,SomeOtherColumn\n'
                        b'"What is the project about?","ExtraValue1"\n'
                        b'"How many components are there?","ExtraValue2"\n'))  # Test file upload mode
])
def test_validate_competency_questions(use_default_dataset, file_upload):
    data = {
        "validation_mode": "all",
        "output_folder": "heatmaps",
        "use_default_dataset": str(use_default_dataset),
        "external_service_url": "http://127.0.0.1:8001/newapi",
        "api_key": "key",
        "model": "gpt-4",
        "save_results": "True"
    }
    files = {}
    if file_upload:
        files = {"file": ("benchmarkdataset.csv", file_upload, "text/csv")}
    response = client.post("/validate/", data=data, files=files)
    assert response.status_code == 200
    json_data = response.json()
    import json
    print(json.dumps(response.json(), indent=2))

    print("Response JSON:", response.json())
    assert "validation_results" in json_data

