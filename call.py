import subprocess
import fire
import dotenv


dotenv.load_dotenv()

"""
TESTED_LLM:
    mistralai/mistral-small-24b-instruct-2501
    # mistralai/mistral-small-3.1-24b-instruct
    mistralai/mistral-small-3.2-24b-instruct
    google/gemma-4-26b-a4b-it

    meta-llama/llama-3-8b-instruct
    meta-llama/llama-3.1-8b-instruct
    mistralai/ministral-8b-2512
"""

TESTED_LLM = "meta-llama/llama-3.1-8b-instruct"
GENERATOR_LLM = "mistralai/mistral-nemo"
GENERATOR_LLM_PROVIDER = "openrouter"


def run_validation(
    target_url:             str        = "http://127.0.0.1:8000/validate/",
    file:                   str | None = None,  # Set None when use_default_dataset is "true"; Maps to file: UploadFile
    validation_mode:        str        = "all",
    output_folder:          str        = "heatmaps",
    use_default_dataset:    str | None = "true",
    external_service_url:   str        = "http://127.0.0.1:8001/newapi",
                                       # "http://127.0.0.1:8010/kg/generate"
    api_key:                str | None = None,  # [?]
    model:                  str | None = TESTED_LLM,
    save_results:           str | None = None,  # "true", 
    save_every:             int | None = None,  # 1,
    evaluator_llm:          str | None = TESTED_LLM,
    tool_llm:               str | None = None,  # [?]
    generator_llm_provider: str | None = GENERATOR_LLM_PROVIDER,
    generator_model:        str | None = GENERATOR_LLM,  # None,
    generated_csv_path:     str | None = None,
):
    """
    A script that executes a curl command mapping ALL parameters to the FastAPI endpoint.
    """

    # Base command
    cmd = ["curl", "-X", "POST", target_url]

    # Handle the File upload (Must use @ syntax in curl)
    if file is not None:
        cmd.extend(["-F", f"file=@{file}"])

    # 2. Build a dictionary of the standard form fields
    form_fields = {
        "validation_mode": validation_mode,
        "output_folder": output_folder,
        "use_default_dataset": use_default_dataset,
        "external_service_url": external_service_url,
        "api_key": api_key,
        "model": model,
        "save_results": save_results,
        "save_every": save_every,
        "evaluator_llm": evaluator_llm,
        "tool_llm": tool_llm,
        "generator_llm_provider": generator_llm_provider,
        "generator_model": generator_model,
        "generated_csv_path": generated_csv_path
    }
    for key, value in form_fields.items():
        if value is not None:
            # We convert everything to a string because curl requires string arguments
            cmd.extend(["-F", f"{key}={str(value)}"])

    # Console Output for debugging
    print("-" * 40)
    print("Executing command:\n")
    # Formatting for terminal print output
    printable_cmd = []
    for x in cmd:
        if x.startswith("-F"):
            printable_cmd.append(x)
        elif "=" in x:
            printable_cmd.append(f'"{x}"')
        else:
            printable_cmd.append(x)
            
    print(" \\\n  ".join(printable_cmd))
    print("-" * 40)
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed with return code: {e.returncode}")
    except FileNotFoundError:
        print("\n[ERROR] 'curl' command not found in the system.")


if __name__ == "__main__":
    """
    to run:
        python3 call.py --parameter=value
    e.g
        python3 call.py --external_service_url=https://huggingface.co/spaces/b289zhan/OntoChat 
    """
    fire.Fire(run_validation)