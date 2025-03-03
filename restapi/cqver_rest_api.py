from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import pandas as pd
import json
import openai
from io import StringIO
from cq_verification import validate_cq  # Make sure you update validate_cq as described above.
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import re

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
        dataset_description: str,
        sample: Optional[str] = None,
        generation_mode: str = "dataset+description",
        user_stories: Optional[str] = None,
        model: str = "gpt-4"  # New model parameter with default "gpt-4"
) -> str:
    """
    Generates competency questions from a DataFrame based on the selected scenario.

    Parameters:
      - dataset_description: description of the dataset.
      - sample: Optional custom dataset sample; if not provided, the first 10 rows are used.
      - generation_mode: One of:
           "user_stories"         -> use user stories,
           "dataset"              -> use dataset sample only,
           "dataset+description"  -> use both dataset description and sample.
      - user_stories: Only used when generation_mode is "user_stories".
      - model: Optional model name for OpenAI API (default is "gpt-4").

    Returns:
      The generated competency questions.
    """
    # Build prompt context based on the selected generation mode
    if generation_mode == "user_stories":
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
            "Invalid generation mode specified. Please choose from 'user_stories', 'dataset', or 'dataset+description'.")

    # Define patterns and instructions (same for all scenarios)
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

    # Build the complete messages for OpenAI
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
        model=model,  # Use the provided model
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
        model: str = Form("gpt-4")  # New optional model parameter (default: "gpt-4")
):
    """
    Endpoint to generate competency questions.

    Scenarios:
      - If generation_mode is "user_stories", the 'user_stories' field is used.
      - If generation_mode is "dataset", only a dataset sample (either auto-extracted or provided)
        is used.
      - If generation_mode is "dataset+description", both the dataset description and sample are used.

    You can also specify an optional model name (default: "gpt-4").

    Example cURL for user stories:

    curl -X POST "http://127.0.0.1:8000/generate" \
      -F "file=@/path/to/your/dataset.csv" \
      -F "generation_mode=user_stories" \
      -F "user_stories=User story 1. User story 2." \
      -F "api_key=YOUR_OPENAI_API_KEY" \
      -F "model=gpt-4"
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
            model=model  # Pass the model parameter along
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
        model: str = Form("gpt-4")  # New optional model parameter for validation (default: "gpt-4")
):
    """
    Endpoint to validate competency questions.
    Expects a CSV file upload that contains both 'gold standard' and 'generated' columns.
    You can optionally specify a model name.
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
            # Pass the model parameter to validate_cq
            result_text = validate_cq(input_text, mode=validation_mode, output_folder=output_folder, model=model)
            # Remove HTML tags from the result_text before adding it to the response.
            clean_result = remove_html_tags(result_text)
            results.append({"row": idx, "result": clean_result})
        except Exception as e:
            results.append({"row": idx, "result": f"Error: {str(e)}"})

    return JSONResponse(content={"validation_results": results})


'''
Example of usage:

1. Generation from user stories:

curl -X POST "http://127.0.0.1:8000/generate" \
  -F "file=@/path/to/your/dataset.csv" \
  -F "generation_mode=user_stories" \
  -F "user_stories=User story 1. User story 2." \
  -F "api_key=YOUR_OPENAI_API_KEY" \
  -F "model=gpt-4"

2. Generation from dataset only:

curl -X POST "http://127.0.0.1:8000/generate" \
  -F "file=@/path/to/your/dataset.csv" \
  -F "generation_mode=dataset" \
  -F "dataset_sample=Optional custom sample text" \
  -F "api_key=YOUR_OPENAI_API_KEY" \
  -F "model=gpt-4"

3. Generation from dataset + description:

curl -X POST "http://127.0.0.1:8000/generate" \
  -F "file=@/path/to/your/dataset.csv" \
  -F "dataset_description=This is a sample dataset description" \
  -F "generation_mode=dataset+description" \
  -F "dataset_sample=Optional custom sample text" \
  -F "api_key=YOUR_OPENAI_API_KEY" \
  -F "model=gpt-4"

For validation:

curl -X POST "http://127.0.0.1:8000/validate" \
  -F "file=@/path/to/your/dataset.csv" \
  -F "validation_mode=all" \
  -F "output_folder=my_heatmaps" \
  -F "api_key=YOUR_OPENAI_API_KEY" \
  -F "model=gpt-4"
'''
