import requests
import pandas as pd

url = "http://127.0.0.1:8001/newapi/"
csv_path = "restapi/app/benchmarkdataset.csv"

temperature_vals = [0, 0.7]
max_token_vals = [50, 100]
presence_penalty_vals = [0.0, 0.7]

results = []

for temp in temperature_vals:
    for max_tok in max_token_vals:
        for presence_penalty in presence_penalty_vals:
            print(f"Testing T={temp}, MaxT={max_tok}, PresencePenalty={presence_penalty}")
            with open(csv_path, "rb") as f:
                files = {"file": ("input.csv", f, "text/csv")}
                data = {
                    "llm_provider": "together",
                    "temperature": str(temp),
                    "max_tokens": str(max_tok),
                    "presence_penalty": str(presence_penalty),
                }
                resp = requests.post(url, files=files, data=data)
                if resp.status_code == 200:
                    results.append({
                        "temperature": temp,
                        "max_tokens": max_tok,
                        "presence_penalty": presence_penalty,
                        "response_csv": resp.text
                    })
                else:
                    results.append({
                        "temperature": temp,
                        "max_tokens": max_tok,
                        "presence_penalty": presence_penalty,
                        "response_csv": f"Error {resp.status_code}"
                    })

df_results = pd.DataFrame(results)

df_results["response_csv"] = df_results["response_csv"].apply(lambda x: '"' + x.replace('"', '""') + '"')
df_results.to_csv("test_combinations_results.csv", index=False, encoding="utf-8")

print("Test results saved to test_results.csv")
