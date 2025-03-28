import pandas as pd
import requests
from io import StringIO

def call_external_cq_generation_service(df: pd.DataFrame, external_service_url: str) -> pd.DataFrame:
    """
    Calls an external CQ generation service to add a 'generated' column to the dataframe.
    """
    csv_data = df.to_csv(index=False)
    files = {
        "file": ("benchmarkdataset.csv", csv_data, "text/csv")
    }
    try:
        response = requests.post(external_service_url, files=files)
    except Exception as e:
        raise Exception(f"Error calling external CQ generation service: {e}")

    if response.status_code != 200:
        raise Exception(f"External CQ generation service error: {response.text}")

    try:
        df_generated = pd.read_csv(StringIO(response.text))
    except Exception as e:
        raise Exception(f"Error reading response CSV from external service: {e}")

    if "generated" not in df_generated.columns:
        raise Exception("The external service did not return a 'generated' column.")

    df["generated"] = df_generated["generated"]
    return df
