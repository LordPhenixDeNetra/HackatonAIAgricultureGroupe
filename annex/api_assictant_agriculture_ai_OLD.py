from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import json
import os
from datetime import datetime

# Import de la classe VirtualAgricultureAssistance
from assictant_agriculture_ai.py import VirtualAgricultureAssistance

# Modèles Pydantic pour la validation des données
class TranscriptionResponse(BaseModel):
    text: str
    language: str
    timestamp: str

class NemotronResponse(BaseModel):
    response: str
    timestamp: str

class NemotronRequest(BaseModel):
    text: str
    temperature: Optional[float] = 0.5
    max_tokens: Optional[int] = 1024

# Initialisation de l'application FastAPI
app = FastAPI(
    title="API Assistant Virtuel Agricole",
    description="API pour la transcription audio et l'interaction avec le modèle Nemotron",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialisation de l'assistant
API_KEY = "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"
assistant = VirtualAgricultureAssistance(api_key=API_KEY)

# Dossier pour stocker les fichiers audio temporaires
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Endpoint pour transcrire un fichier audio.
    
    Args:
        file (UploadFile): Fichier audio à transcrire
        
    Returns:
        TranscriptionResponse: Résultat de la transcription
    """
    try:
        # Sauvegarde temporaire du fichier
        temp_path = os.path.join(UPLOAD_DIR, f"{datetime.now().timestamp()}_{file.filename}")
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Transcription
        result = assistant.transcribe_audio(temp_path)
        
        # Nettoyage
        os.remove(temp_path)
        
        return TranscriptionResponse(
            text=result["text"],
            language=result["language"],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class StreamResponse:
    def __init__(self):
        self.content = []

    def append(self, text):
        self.content.append(text)

@app.post("/query-nemotron", response_model=NemotronResponse)
async def query_nemotron(request: NemotronRequest):
    """
    Endpoint pour interroger le modèle Nemotron.
    
    Args:
        request (NemotronRequest): Requête pour le modèle Nemotron
        
    Returns:
        NemotronResponse: Réponse du modèle
    """
    try:
        # Création d'un objet pour collecter la réponse en streaming
        stream_response = StreamResponse()
        
        # Interrogation du modèle
        completion = assistant.client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-70b-instruct",
            messages=[{"role": "user", "content": request.text}],
            temperature=request.temperature,
            top_p=1,
            max_tokens=request.max_tokens,
            stream=True
        )
        
        # Collecte de la réponse
        full_response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
        
        return NemotronResponse(
            response=full_response,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": "Bienvenue sur l'API de l'Assistant Virtuel Agricole",
        "version": "1.0.0",
        "endpoints": {
            "/transcribe": "POST - Transcription audio",
            "/query-nemotron": "POST - Interrogation du modèle Nemotron",
        }
    }

def start_server():
    """Démarrage du serveur"""
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    start_server()