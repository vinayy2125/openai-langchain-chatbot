import requests
from langchain.memory import ConversationBufferMemory
from datetime import datetime
import os
import openai
import re


def web_search(query: str) -> str:
    """
    Perform a simple web search using DuckDuckGo Instant Answer API.
    Returns a summary or snippet.
    """
    url = f"https://api.duckduckgo.com/?q={query}&format=json"
    try:
        resp = requests.get(url)
        data = resp.json()
        return data.get("AbstractText") or "No summary found."
    except Exception as e:
        return f"Web search error: {e}"


def calculator(expression: str) -> str:
    """
    Evaluate a simple math expression safely.
    """
    try:
        # Only allow safe built-in functions
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Calculator error: {e}"


def current_datetime(_: str = "") -> str:
    """
    Return the current date and time as a formatted string.
    """
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y %H:%M:%S")


def image_generation(prompt: str) -> str:
    print(f"[DEBUG] image_generation tool called with prompt: {prompt}")
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.images.generate(
            model="dall-e-3", prompt=prompt, n=1, size="1024x1024"
        )
        image_url = response.data[0].url
        print(f"[DEBUG] Got image URL: {image_url}")

        # Download image
        img_response = requests.get(image_url)
        if img_response.status_code == 200:
            filename = f"generated_images/{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
            os.makedirs("generated_images", exist_ok=True)
            with open(filename, "wb") as f:
                f.write(img_response.content)
            print(f"[DEBUG] Image saved to: {filename}")
            return filename  # Return local path
        else:
            return "Failed to fetch image from OpenAI."

    except Exception as e:
        print(f"[DEBUG] DALL·E error: {e}")
        return f"Image generation error: {e}"

