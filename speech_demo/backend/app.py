from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration
)

from peft import PeftModel

from jiwer import wer

import librosa
import tempfile
import torch

# ======================================================
# CONFIG
# ======================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_MODEL_PATH = "../models/whisper-base"

LORA_MODELS = {
    "sports": "../models/sports_lora",
    "weather": "../models/weather_lora",
    "music": "../models/music_lora"
}

# ======================================================
# APP
# ======================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# LOAD PROCESSOR
# ======================================================

processor = WhisperProcessor.from_pretrained(
    BASE_MODEL_PATH
)

# ======================================================
# LOAD BASE MODEL
# ======================================================

base_model = WhisperForConditionalGeneration.from_pretrained(
    BASE_MODEL_PATH
).to(DEVICE)

base_model.eval()

# ======================================================
# LOAD LORA MODELS
# ======================================================

lora_models = {}

for domain, path in LORA_MODELS.items():

    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH
    )

    model = PeftModel.from_pretrained(
        model,
        path
    )

    model = model.to(DEVICE)
    model.eval()

    lora_models[domain] = model

print("All models loaded.")

# ======================================================
# NORMALIZATION
# ======================================================

def normalize(text):

    return text.lower().strip()

# ======================================================
# TRANSCRIBE
# ======================================================

def transcribe(model, audio):

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    input_features = inputs.input_features.to(DEVICE)

    with torch.no_grad():

        predicted_ids = model.generate(
            input_features,
            language="en",
            task="transcribe"
        )

    text = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    return normalize(text)

# ======================================================
# API
# ======================================================

@app.post("/evaluate")

async def evaluate(
    file: UploadFile = File(...),
    domain: str = Form(...),
    reference: str = Form(...)
):

    # save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:

        tmp.write(await file.read())

        temp_path = tmp.name

    # load audio
    audio, sr = librosa.load(temp_path, sr=16000)

    # ==================================================
    # BASE TRANSCRIPTION
    # ==================================================

    base_text = transcribe(base_model, audio)

    # ==================================================
    # LORA TRANSCRIPTION
    # ==================================================

    lora_model = lora_models[domain]

    lora_text = transcribe(lora_model, audio)

    # ==================================================
    # WER
    # ==================================================

    ref = normalize(reference)

    base_wer = wer(ref, base_text) * 100
    lora_wer = wer(ref, lora_text) * 100

    improvement = base_wer - lora_wer

    return {
        "domain": domain,

        "reference": ref,

        "base_transcript": base_text,
        "lora_transcript": lora_text,

        "base_wer": round(base_wer, 2),
        "lora_wer": round(lora_wer, 2),

        "improvement": round(improvement, 2)
    }