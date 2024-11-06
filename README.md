# Assistant Virtuel d'Agriculture 🌾

## Description
L'Assistant Virtuel d'Agriculture est une API intelligente qui combine la reconnaissance vocale en langues locales (notamment le Wolof) avec des capacités avancées de traitement du langage naturel pour assister les agriculteurs. Le système utilise le modèle Whisper-Wolof pour la transcription audio et le modèle Nemotron 70B pour générer des réponses pertinentes aux questions des agriculteurs.

## Table des matières
- [Fonctionnalités](#fonctionnalités)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Configuration](#configuration)
- [Utilisation de l'API](#utilisation-de-lapi)
- [Exemples de Requêtes](#exemples-de-requêtes)
<!-- - [Structure du Projet](#structure-du-projet) -->
- [Dépendances](#dépendances)
- [Contribution](#contribution)
- [Licence](#licence)

## Fonctionnalités
- 🎤 Enregistrement audio en temps réel
- 🗣️ Transcription audio multilingue (focus sur le Wolof)
- 💬 Traitement des requêtes agricoles via IA
- 🌐 API RESTful pour l'intégration facile
- 📱 Support de différents formats audio (WAV, MP3, OGG, M4A)

## Prérequis
- Python 3.8+
- pip (gestionnaire de paquets Python)
- Clé API NVIDIA
- Microphone (pour l'enregistrement audio)
- Flask
- PyTorch
- Transformers
- sounddevice
- OpenAI

## Installation

1. Clonez le repository :
```bash
git clone https://github.com/LordPhenixDeNetra/HackatonAIAgricultureGroupe

cd HackatonAIAgricultureGroupe
```

2. Créez un environnement virtuel :
```bash
python -m venv env
source env/bin/activate  # Unix/macOS
# ou
.\env\Scripts\activate  # Windows
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Configuration

1. Définissez votre clé API NVIDIA :
```python
API_KEY = "votre-clé-api-nvidia"
```

2. Configurez les paramètres audio si nécessaire :
```python
SAMPLE_RATE = 16000
CHANNELS = 1
```

## Utilisation de l'API

L'API expose quatre endpoints principaux :

### 1. Vérification de Santé
```http
GET /health
```

### 2. Transcription Audio
```http
POST /api/transcribe
Content-Type: multipart/form-data

file: <audio-file>
```

### 3. Enregistrement Audio
```http
POST /api/record
Content-Type: application/json

{
    "duration": 5
}
```

### 4. Requête au Modèle
```http
POST /api/query
Content-Type: application/json

{
    "prompt": "Votre question ici",
    "temperature": 0.7,
    "max_tokens": 1024
}
```

## Exemples de Requêtes

### Avec cURL

```bash
# Vérification de santé
curl http://localhost:5000/health

# Transcription d'un fichier audio
curl -X POST -F "file=@audio.wav" http://localhost:5000/api/transcribe

# Enregistrement audio
curl -X POST -H "Content-Type: application/json" \
     -d '{"duration": 5}' \
     http://localhost:5000/api/record

# Requête au modèle
curl -X POST -H "Content-Type: application/json" \
     -d '{"prompt": "Quels sont les meilleurs moments pour planter du maïs?"}' \
     http://localhost:5000/api/query
```

### Avec Postman

1. Importez la collection Postman fournie
2. Configurez l'environnement local (`localhost:5000`)
3. Testez chaque endpoint avec les exemples prédéfinis

<!-- ## Structure du Projet -->
<!-- ```
assistant-agriculture/
├── api/
│   ├── __init__.py
│   └── routes.py
├── models/
│   └── virtual_agriculture_assistance.py
├── utils/
│   └── audio_processing.py
├── tests/
│   └── test_api.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
``` -->

## Dépendances

Principales dépendances du projet :
```txt
flask==3.0.0
torch==2.1.0
torchaudio==2.1.0
transformers==4.35.0
openai==1.3.0
sounddevice==0.4.6
pydub==0.25.1
scipy==1.11.3
numpy==1.24.3
```


## Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE.md](LICENSE.md) pour plus de détails.

## Support

Pour toute question ou problème :
- 📧 Email : votre-email@exemple.com
- 🐛 Issues : https://github.com/votre-username/assistant-agriculture/issues
- 💬 Discussions : https://github.com/votre-username/assistant-agriculture/discussions

---
Développé avec ❤️ pour soutenir l'agriculture locale