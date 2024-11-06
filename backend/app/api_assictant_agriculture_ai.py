from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import tempfile
from datetime import datetime

# Import de la classe existante
from assictant_agriculture_ai import VirtualAgricultureAssistance

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = tempfile.gettempdir()  # Dossier temporaire pour les fichiers audio
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a'}
API_KEY = "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"  # À mettre dans les variables d'environnement

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
assistant = VirtualAgricultureAssistance(api_key=API_KEY)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de vérification de l'état de l'API"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    """Endpoint pour transcrire un fichier audio"""
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier n'a été envoyé"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Aucun fichier sélectionné"}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Sauvegarde temporaire du fichier
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Transcription
            result = assistant.transcribe_audio(filepath)
            
            # Nettoyage
            os.remove(filepath)
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "Type de fichier non autorisé"}), 400

@app.route('/api/record', methods=['POST'])
def record_audio():
    """Endpoint pour enregistrer de l'audio"""
    try:
        data = request.get_json()
        duration = data.get('duration', 5)
        
        filename = f"record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        result = assistant.record_and_transcribe(
            duration=duration,
            output_file=filepath,
            delete_after=True
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/query', methods=['POST'])
def query_nemotron():
    """Endpoint pour interroger le modèle Nemotron"""
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({"error": "Le prompt est requis"}), 400
        
        prompt = data['prompt']
        temperature = data.get('temperature', 0.5)
        max_tokens = data.get('max_tokens', 1024)
        
        response = assistant.query_nemotron_text(prompt)
        
        return jsonify({
            "prompt": prompt,
            "response": response
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Route non trouvée"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Erreur interne du serveur"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)