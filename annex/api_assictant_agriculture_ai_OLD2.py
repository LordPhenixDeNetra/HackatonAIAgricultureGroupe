from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel
from typing import Optional
import json
import os
from datetime import datetime

# Import de la classe VirtualAgricultureAssistance
from assictant_agriculture_ai import VirtualAgricultureAssistance

app = Flask(__name__)
CORS(app)

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

# Initialisation de l'assistant
API_KEY = "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"
assistant = VirtualAgricultureAssistance(api_key=API_KEY)

# Dossier pour stocker les fichiers audio temporaires
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """
    Endpoint pour transcrire un fichier audio.
    """
    try:
        file = request.files.get('file')
        if file is None:
            return jsonify({"error": "File is required"}), 400
        
        # Sauvegarde temporaire du fichier
        temp_path = os.path.join(UPLOAD_DIR, f"{datetime.now().timestamp()}_{file.filename}")
        file.save(temp_path)
        
        # Transcription
        result = assistant.transcribe_audio(temp_path)
        
        # Nettoyage
        os.remove(temp_path)
        
        # Formatage de la réponse
        response = TranscriptionResponse(
            text=result["text"],
            language=result["language"],
            timestamp=datetime.now().isoformat()
        ).dict()
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/query-nemotron', methods=['POST'])
def query_nemotron():
    """
    Endpoint pour interroger le modèle Nemotron.
    """
    try:
        data = request.get_json()
        nemotron_request = NemotronRequest(**data)
        
        # Création d'un objet pour collecter la réponse en streaming
        full_response = ""
        
        # Interrogation du modèle
        completion = assistant.client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-70b-instruct",
            messages=[{"role": "user", "content": nemotron_request.text}],
            temperature=nemotron_request.temperature,
            top_p=1,
            max_tokens=nemotron_request.max_tokens,
            stream=True
        )
        
        # Collecte de la réponse
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
        
        response = NemotronResponse(
            response=full_response,
            timestamp=datetime.now().isoformat()
        ).dict()
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/query-nemotron-text', methods=['POST'])
def query_nemotron_text():
    """
    Endpoint pour interroger le modèle Nemotron avec un prompt en texte.
    """
    try:
        # Récupérer les données envoyées
        data = request.get_json()
        print("Données reçues:", data)  # Affiche les données reçues
        
        text = data.get("text")
        temperature = data.get("temperature", 0.5)
        max_tokens = data.get("max_tokens", 1024)
        
        if not text:
            return jsonify({"error": "Le champ 'text' est requis"}), 400

        # Interrogation du modèle
        print("Interrogation du modèle avec les paramètres:", text, temperature, max_tokens)

        completion = assistant.client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-70b-instruct",
            messages=[{"role": "user", "content": text}],
            temperature=temperature,
            top_p=1,
            max_tokens=max_tokens,
            stream=True
        )
        
        # Afficher la réponse brute du modèle pour vérifier sa structure
        print("Réponse brute du modèle:", completion)

        # Initialiser la réponse complète
        full_response = ""
        
        # Collecter la réponse
        for chunk in completion:
            print("Chunk reçu:", chunk)  # Affiche chaque chunk reçu
            
            # Vérifier la structure de chaque chunk
            if isinstance(chunk, dict) and "choices" in chunk:
                choice = chunk["choices"][0]
                print("Choice:", choice)  # Affiche le choix
                
                if "delta" in choice and choice["delta"].get("content"):
                    full_response += choice["delta"]["content"]
        
        # Vérifier si nous avons obtenu une réponse complète
        if not full_response:
            return jsonify({"error": "Aucune réponse générée"}), 500
        
        return jsonify({
            "response": full_response,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        # Capturer l'exception complète et l'afficher dans les logs
        print("Erreur:", str(e))
        return jsonify({"error": str(e)}), 500



@app.route('/')
def root():
    """Page d'accueil de l'API"""
    return jsonify({
        "message": "Bienvenue sur l'API de l'Assistant Virtuel Agricole",
        "version": "1.0.0",
        "endpoints": {
            "/transcribe": "POST - Transcription audio",
            "/query-nemotron": "POST - Interrogation du modèle Nemotron",
            "/query-nemotron-text": "POST - Interrogation du modèle Nemotron",
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
