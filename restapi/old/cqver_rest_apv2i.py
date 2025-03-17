from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import pandas as pd
import json
import openai
from io import StringIO
from cq_verification import validate_cq  # Make sure you update validate_cq as needed.
from typing import Optional
import matplotlib

matplotlib.use("Agg")
import re
import logging

logging.basicConfig(
    level=logging.DEBUG,  # set to DEBUG for more detailed logs
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)



def remove_html_tags(text: str) -> str:
    """Remove HTML tags from a string."""
    return re.sub(r'<[^>]+>', '', text)


app = FastAPI(
    title="Competency Question API",
    description="API to generate and validate competency questions based on CSV datasets.",
    version="1.0.0"
)


def generate_cq_from_df(
        df: pd.DataFrame,
        dataset_description: str = "",
        sample: Optional[str] = None,
        generation_mode: str = "dataset+description",
        user_stories: Optional[str] = None,
        model: str = "gpt-4"
) -> str:
    """
    Generates competency questions from a DataFrame based on the selected scenario.

    Supported modes:
      - "user_stories": Uses user stories.
      - "dataset": Uses dataset sample only.
      - "dataset+description": Uses both dataset description and sample.
      - "scenario": Uses the Scenario column when Description and Dataset are empty.
    """
    if generation_mode == "scenario":
        # Check for the 'Scenario' column
        if "Scenario" not in df.columns:
            raise ValueError("CSV file must have a 'Scenario' column for scenario mode.")
        # Ensure Description and Dataset columns are empty (or not present)
        description_empty = ("Description" not in df.columns or
                             df["Description"].dropna().apply(lambda x: str(x).strip() == "").all())
        dataset_empty = ("Dataset" not in df.columns or
                         df["Dataset"].dropna().apply(lambda x: str(x).strip() == "").all())
        if not (description_empty and dataset_empty):
            raise ValueError("For scenario mode, both 'Description' and 'Dataset' must be empty.")
        # Extract non-empty scenarios
        scenario_series = df["Scenario"].dropna().apply(lambda x: str(x).strip())
        if scenario_series.empty:
            raise ValueError("No scenario data available for scenario mode.")
        unique_scenarios = scenario_series.unique()
        scenario_prompt = "\n".join(unique_scenarios)
        prompt_context = f"Scenario(s):\n{scenario_prompt}"

    elif generation_mode == "user_stories":
        if not user_stories:
            raise ValueError("User stories must be provided for generation_mode 'user_stories'.")
        prompt_context = f"User stories: {user_stories}"

    elif generation_mode == "dataset":
        if sample:
            dataset_sample = sample
        else:
            snippet_df = df.head(10)
            common_scenario = None
            if "scenario" in snippet_df.columns:
                scenarios = snippet_df["scenario"].unique()
                if len(scenarios) == 1:
                    common_scenario = scenarios[0]
                    snippet_df = snippet_df.drop("scenario", axis=1)
            dataset_sample = snippet_df.to_string(index=False)
            if common_scenario:
                dataset_sample += f"\nScenario: {common_scenario}"
        prompt_context = f"Dataset sample: {dataset_sample}"

    elif generation_mode == "dataset+description":
        if sample:
            dataset_sample = sample
        else:
            snippet_df = df.head(10)
            common_scenario = None
            if "scenario" in snippet_df.columns:
                scenarios = snippet_df["scenario"].unique()
                if len(scenarios) == 1:
                    common_scenario = scenarios[0]
                    snippet_df = snippet_df.drop("scenario", axis=1)
            dataset_sample = snippet_df.to_string(index=False)
            if common_scenario:
                dataset_sample += f"\nScenario: {common_scenario}"
        prompt_context = f"Dataset description: {dataset_description}\n\nDataset sample: {dataset_sample}"

    else:
        raise ValueError(
            "Invalid generation mode specified. Please choose from 'user_stories', 'dataset', 'dataset+description', or 'scenario'."
        )

    # Define patterns and instructions for generating competency questions.
    patterns = [
        {"pattern": "Which [class expression 1][object property expression][class expression 2]?",
         "example": "Which pizzas contain pork?"},
        {"pattern": "How much does [class expression][datatype property]?",
         "example": "How much does Margherita Pizza weigh?"},
        {"pattern": "What type of [class expression] is [individual]?",
         "example": "What type of software (API, Desktop application etc.) is it?"},
        {"pattern": "Is the [class expression 1][class expression 2]?",
         "example": "Is the software open source development?"},
        {"pattern": "What [class expression] has the [numeric modifier][datatype property]?",
         "example": "What pizza has the lowest price?"},
        {"pattern": "Which are [class expressions]?",
         "example": "Which are gluten-free bases?"}
    ]
    instructions = [
        {
            "instruction": "Do not make explicit references to the dataset or its variables in the generated competency questions.",
            "example": {
                "incorrect": "How many cases of Salmonella were reported in Lombardy in 2020?",
                "correct": "How many cases of the disease were reported in the region in a given year?"
            }
        },
        {
            "instruction": "Keep the questions simple. Each competency question should not contain another simpler competency question within it.",
            "example": {
                "incorrect": "Who wrote The Hobbit and in what year was the book written?",
                "correct": ["Who wrote the book?", "In what year was the book written?"]
            }
        },
        {
            "instruction": "Do not include real entities; instead, abstract them into more generic concepts.",
            "example": {
                "incorrect": "Who is the author of 'Harry Potter'?",
                "correct": "Who is the author of the book?"
            }
        }
    ]
    clustering_instructions = (
        "Once the competency questions have been generated, they should be clustered into thematic areas. "
        "Each cluster represents an ontological module in the format: area : competency question and separated by ; . "
        "For example: Doctoral Theses Analysis : Which departments had new enrollments in a specific year?; "
        "Doctoral Theses Analysis : How many unique departments are listed in the dataset?;"
    )

    messages = [
        {"role": "system",
         "content": (
             "You are an ontology engineer. Generate a list of competency questions based on the provided context, "
             "following these patterns and instructions. Use the following competency question patterns:\n"
             f"{json.dumps(patterns)}\n\n"
             "Follow these instructions when generating the competency questions:\n"
             f"{json.dumps(instructions, indent=2)}\n\n"
             "After generating the questions, cluster them into thematic areas according to these guidelines:\n"
             f"{clustering_instructions}"
         )},
        {"role": "user", "content": prompt_context}
    ]

    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=4000,
        temperature=0
    )
    generated_cq = response.choices[0].message.content.strip()
    return generated_cq


@app.post("/generate")
async def generate_competency_questions(
        file: UploadFile = File(...),
        dataset_description: str = Form(""),
        dataset_sample: Optional[str] = Form(None),
        generation_mode: str = Form("dataset+description"),
        user_stories: Optional[str] = Form(None),
        api_key: Optional[str] = Form(None),
        model: str = Form("gpt-4")
):
    """
    Endpoint to generate competency questions.
    Modes:
      - "user_stories": Uses user stories.
      - "dataset": Uses dataset sample.
      - "dataset+description": Uses both description and dataset.
      - "scenario": Uses Scenario column when Description and Dataset are empty.
    """
    if api_key:
        openai.api_key = api_key

    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV file: {str(e)}")

    try:
        generated_cq = generate_cq_from_df(
            df,
            dataset_description,
            sample=dataset_sample,
            generation_mode=generation_mode,
            user_stories=user_stories,
            model=model
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating competency questions: {str(e)}")

    return JSONResponse(content={"generated_competency_questions": generated_cq})


@app.post("/validate")
async def validate_competency_questions(
        file: UploadFile = File(...),
        validation_mode: str = Form("all"),
        output_folder: str = Form("heatmaps"),
        api_key: Optional[str] = Form(None),
        model: str = Form("gpt-4")
):
    """
    Endpoint to validate competency questions.
    Expects a CSV file containing both 'gold standard' and 'generated' columns.
    """
    if api_key:
        openai.api_key = api_key

    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV file: {str(e)}")

    if "gold standard" not in df.columns or "generated" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="CSV file must contain 'gold standard' and 'generated' columns for validation."
        )

    results = []
    for idx, row in df.iterrows():
        input_text = f"Gold standard: {row['gold standard']}\nGenerated: {row['generated']}"
        try:
            result_text = validate_cq(input_text, mode=validation_mode, output_folder=output_folder, model=model)
            clean_result = remove_html_tags(result_text)
            results.append({"row": idx, "result": clean_result})
        except Exception as e:
            results.append({"row": idx, "result": f"Error: {str(e)}"})

    return JSONResponse(content={"validation_results": results})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
