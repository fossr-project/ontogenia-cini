from fastapi import FastAPI, File, UploadFile, HTTPException, Response
import pandas as pd
import openai
import logging
from io import StringIO
import os

openai.api_key = "mykey"

# Logging setup
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CQ Generator Service",
    description="Standalone API to generate competency questions from a benchmark CSV.",
    version="1.0.0"
)

@app.post("/newapi/")
async def generate_cqs_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
        df = df.head(5)  #CHANGE IF YOU WANT TO MODIFY THE FULL DATASET
    except Exception as e:
        logger.error(f"CSV read error: {e}")
        raise HTTPException(status_code=400, detail=f"Error reading CSV file: {e}")

    if "gold standard" in df.columns:
        gold_col = "gold standard"
    elif "Competency Question" in df.columns:
        gold_col = "Competency Question"
    else:
        raise HTTPException(status_code=400, detail="CSV must contain 'gold standard' or 'Competency Question' column.")

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
            logger.error(f"OpenAI error for question '{gold_question}': {e}")
            generated = f"Error generating CQ: {e}"

        generated_questions.append(generated)

    df["generated"] = generated_questions
    csv_output = df.to_csv(index=False)
    return Response(content=csv_output, media_type="text/csv")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("cq_generator_app:app", host="127.0.0.1", port=8001, reload=True)
