import base64
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

app = FastAPI(title="Gemini Image Analyzer API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for dev; tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Function (async) ---
async def encode_image_file(file: UploadFile):
    """
    Encodes uploaded image to base64 and returns (encoded_string, mime_type).
    Uses UploadFile.read() which is async-safe.
    """
    try:
        mime_type = file.content_type
        if not mime_type or not mime_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

        # Use async read provided by FastAPI's UploadFile
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        encoded_string = base64.b64encode(file_bytes).decode("utf-8")
        return encoded_string, mime_type

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")


system_prompt = """
You are a helpful AI Assistant specialized in analyzing images.
The user provides images of only two categories:
1. Food
2. Medical Prescription

If the image is of food:
1. List names of food items.
2. Provide calories for each item based on quantity.
3. Calculate total calories.

If the image is of a medical prescription:
1. List medicine names.
2. Describe benefits.
3. Provide dosage.
4. Estimate medicine prices.
5. Suggest verified, cheaper generic alternatives if available.

Rules:
- Accuracy should be about 90%.
- Suggest generics only if verified by a government source.
- Keep dosage clear and language simple.

Return the result as JSON with a single key 'Conclusion' containing the full analysis.
"""

@app.post("/analysis")
async def analyze_image(file: UploadFile = File(...)):
    """
    Upload an image for AI analysis.
    POST /analysis
    """
    try:
        encoded_image, mime_type = await encode_image_file(file)

        # Build message content — keep it simple: system prompt and a user instruction.
        # Put the image as a data URI in the user content.
        user_text = (
            "Analyze this image according to the system prompt's instructions and "
            "provide the result as JSON with a top-level key 'Conclusion'."
        )

        # Messages structure may depend on the SDK; keep same general shape you had.
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps({
                    "note": user_text,
                    "image_data_uri": f"data:{mime_type};base64,{encoded_image}"
                })
            },
        ]

        # Call the client — wrap in try/except and return helpful debug if things go wrong
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            response_format={"type": "json_object"},
            temperature=0.1,
            messages=messages
        )

        # Attempt to extract text content (structure depends on SDK/response)
        try:
            # this is what your code expected
            response_content = response.choices[0].message.content
        except Exception:
            # fallback: stringify response for debugging
            response_content = getattr(response, "text", None) or json.dumps(response, default=str)

        # Try parse as JSON
        try:
            parsed_output = json.loads(response_content)
        except Exception:
            # If the model didn't return strict JSON, return raw text for debugging
            return JSONResponse(
                status_code=502,
                content={
                    "status": "error",
                    "detail": "AI response was not valid JSON.",
                    "ai_raw": response_content
                }
            )

        if "Conclusion" in parsed_output:
            return JSONResponse(content={"status": "success", "data": parsed_output["Conclusion"]})
        else:
            return JSONResponse(content={"status": "partial", "raw": parsed_output})

    except HTTPException as he:
        # re-raise HTTPExceptions with their message/status
        raise he
    except Exception as e:
        # log / return helpful message
        raise HTTPException(status_code=500, detail=f"Server error: {e}")


@app.get("/")
def home():
    return {"message": "Welcome to Gemini Image Analyzer API! Use POST /analysis to upload an image."}
