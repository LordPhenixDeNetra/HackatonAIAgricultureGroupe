# API Assistant Virtuel Agricole

Cette API permet la transcription d'audio en texte en Wolof et interagit avec le modèle Nemotron pour répondre à des requêtes textuelles.

## Structure

- `app/main.py` : Point d'entrée principal de l'API.
- `app/api/v1/routes.py` : Routes de l'API.
- `app/models/schemas.py` : Modèles de données.
- `app/services/assistant_service.py` : Logique métier de l'assistant virtuel.
- `app/core/config.py` : Configuration de l'application.

## Installation

1. Cloner le dépôt.
2. Installer les dépendances :

   ```bash
   pip install -r requirements.txt
