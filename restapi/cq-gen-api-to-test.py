from fastapi import FastAPI, File, UploadFile, HTTPException, Response
import pandas as pd
import openai
import logging
from io import StringIO

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

openai.api_key = "key"

app = FastAPI(
    title="CQ Generation API",
    description="An API that generates competency questions based on a provided CSV benchmark. "
                "It expects a CSV with a gold standard column (either 'gold standard' or 'Competency Question') "
                "and returns a CSV with an added 'generated' column.",
    version="1.0.0"
)


@app.post("/newapi")
async def generate_cqs_endpoint(file: UploadFile = File(...)):
    """
    External CQ Generation Service.

    Expects a CSV file containing at least a 'gold standard' or 'Competency Question' column.
    For each gold standard question, the endpoint calls the GPT API to generate a competency question.
    Returns a CSV file with an added 'generated' column.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
        df = df.head(5)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV file: {e}")

    if "gold standard" in df.columns:
        gold_col = "gold standard"
    elif "Competency Question" in df.columns:
        gold_col = "Competency Question"
    else:
        raise HTTPException(
            status_code=400,
            detail="CSV file must contain a 'gold standard' or 'Competency Question' column."
        )

    generated_questions = []
    for question in df[gold_col]:
        gold_question = str(question).strip()
        if not gold_question:
            generated_questions.append("")
            continue

        prompt = (
            f"Generate a competency question that is semantically equivalent to the following:\n"
            f"\"{gold_question}\""
        )
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a competency question generation assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=60,
                temperature=0.5
            )
            generated = response.choices[0].message.content.strip()
        except Exception as e:
            generated = f"Error generating CQ: {e}"
        generated_questions.append(generated)

    df["generated"] = generated_questions
    csv_output = df.to_csv(index=False)
    return Response(content=csv_output, media_type="text/csv")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)
