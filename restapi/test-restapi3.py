import requests

verification_url = "http://127.0.0.1:8000/validate"

# Here, use_default_dataset is set to "True", so the API loads the default dataset (benchmark.csv)
data = {
    "validation_mode": "all",      # Options: "jaccard", "llm", "cosine", or "all"
    "output_folder": "heatmaps",
    "use_default_dataset": "True",
    "external_service_url": "http://127.0.0.1:8001/newapi",
    "api_key": "key",
    "model": "gpt-4"
}

response = requests.post(verification_url, data=data)

print("Verification API Test (using newapi as external generation service with default dataset):")
print(response.json())


#testcase2
import requests
import io

verification_url = "http://127.0.0.1:8000/validate"

#sample csv
csv_content = """gold standard,SomeOtherColumn
"What is the project about?","ExtraValue1"
"How many components are there?","ExtraValue2"
"""

files = {
    "file": ("benchmarkdataset.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")
}

data = {
    "validation_mode": "all",      # Options: "jaccard", "llm", "cosine", or "all"
    "output_folder": "heatmaps",
    "use_default_dataset": "False",
    "external_service_url": "http://127.0.0.1:8001/newapi",
    "api_key": "key",
    "model": "gpt-4"
}

response = requests.post(verification_url, data=data, files=files)
print("Verification API Test (using newapi as external generation service):")
print(response.json())
