import requests
import json
# os module might not be needed if HF_API_TOKEN is completely removed
# import os

# --- Configuration ---
ENDPOINT_URL = "https://izqynpy6ktzegc74.us-east4.gcp.endpoints.huggingface.cloud"
# HF_API_TOKEN is NOT NEEDED for a public endpoint
# HF_API_TOKEN = os.environ.get("HF_API_TOKEN_ENDPOINT")
# if HF_API_TOKEN is None:
# HF_API_TOKEN = "YOUR_HF_API_TOKEN_HERE"

# --- Client-Side Prompt Formatting Helper for MedGemma ---
def format_medgemma_prompt(user_question: str) -> str:
    return f"<start_of_turn>user\n{user_question}<end_of_turn>\n<start_of_turn>model\n"

def generate_text_from_endpoint(question: str,
                                max_new_tokens: int = 150,
                                temperature: float = 0.7,
                                top_p: float = 0.9) -> dict | None:
    if not ENDPOINT_URL:
        print("Error: ENDPOINT_URL is not set.")
        return None

    formatted_prompt = format_medgemma_prompt(question)

    # NO Authorization header needed for public endpoints
    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "return_full_text": False,
            "stop": ["<end_of_turn>"],  # Ensure proper stopping
            "repetition_penalty": 1.1   # Reduce repetition
        },
        "options": {
            "use_cache": False,
            "wait_for_model": True,
            "use_gpu": True
        },
        "stream": False  # Explicitly disable streaming for complete response
    }

    print(f"Sending request to public endpoint: {ENDPOINT_URL}")
    # ... (rest of the try-except block remains the same) ...
    try:
        response = requests.post(ENDPOINT_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        print("Error: The request timed out.")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"Error: HTTP error occurred: {e}")
        print(f"Response status code: {e.response.status_code}")
        try:
            print(f"Error details from server: {e.response.json()}")
        except json.JSONDecodeError:
            print(f"Error details from server (raw): {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error: An unexpected error occurred during the request: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# --- Example Usage ---
if __name__ == "__main__":
    print("Hugging Face Inference Endpoint Client for Text Generation (Public Endpoint)")
    print("--------------------------------------------------------------------------")

    # No warning about HF_API_TOKEN needed for public endpoint

    # Single test run instead of infinite loop
    user_question = 'How to deal with bed sores ? Please advise'

    print("\nGenerating response...")
    api_response = generate_text_from_endpoint(user_question)

    if api_response:
        if isinstance(api_response, list) and len(api_response) > 0 and "generated_text" in api_response[0]:
            generated_text = api_response[0]["generated_text"]
            print("\nMedGemma's Response:")
            print("--------------------")
            print(generated_text)
            print("--------------------")
        else:
            print("\nReceived an unexpected response format from the server:")
            print(json.dumps(api_response, indent=2))
    else:
        print("Failed to get a response from the server.")
    print("\n")