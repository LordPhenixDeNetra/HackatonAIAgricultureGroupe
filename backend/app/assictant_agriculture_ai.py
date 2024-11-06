import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from openai import OpenAI
from transformers import pipeline
import torch
import torchaudio
from pydub import AudioSegment
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os
from typing import Generator, Optional
import time

class VirtualAgricultureAssistance:
    def __init__(self, api_key: str):
        """
        Initialise l'assistant virtuel.
        
        Args:
            api_key (str): Clé API NVIDIA
        """
        self.api_key = api_key
        self.transcriber = self.load_transcriber()
        self.client = self.init_nvidia_client()
        self.sample_rate = 16000
        self.channels = 1

    def init_nvidia_client(self) -> OpenAI:
        """
        Initialise et retourne un client OpenAI configuré pour l'API NVIDIA.
        
        Returns:
            OpenAI: Instance du client configuré
        """
        try:
            return OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=self.api_key
            )
        except Exception as e:
            print(f"❌ Erreur lors de l'initialisation du client NVIDIA : {str(e)}")
            raise

    def load_transcriber(self):
        """Charge le pipeline de transcription avec le modèle Whisper-Wolof"""
        try:
            return pipeline(
                "automatic-speech-recognition",
                model="cibfaye/whisper-wolof",
                chunk_length_s=30,
                batch_size=8,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        except Exception as e:
            print(f"❌ Erreur lors du chargement du transcripteur : {str(e)}")
            raise

    def convert_to_wav(self, file_path: str) -> str:
        """
        Convertit l'audio en format WAV si nécessaire.
        
        Args:
            file_path (str): Chemin du fichier audio
            
        Returns:
            str: Chemin du fichier WAV
        """
        try:
            if not file_path.lower().endswith(".wav"):
                audio = AudioSegment.from_file(file_path)
                wav_path = file_path.rsplit(".", 1)[0] + ".wav"
                audio.export(wav_path, format="wav")
                print(f"✅ Fichier converti en WAV: {wav_path}")
                return wav_path
            return file_path
        except Exception as e:
            print(f"❌ Erreur lors de la conversion en WAV : {str(e)}")
            raise

    def preprocess_audio(self, file_path: str) -> tuple[torch.Tensor, int]:
        """
        Prétraite l'audio en le convertissant en mono et en 16 kHz.
        
        Args:
            file_path (str): Chemin du fichier audio
            
        Returns:
            tuple: (waveform, sample_rate)
        """
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            if sample_rate != self.sample_rate:
                waveform = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, 
                    new_freq=self.sample_rate
                )(waveform)
            
            return waveform, self.sample_rate
        except Exception as e:
            print(f"❌ Erreur lors du prétraitement audio : {str(e)}")
            raise

    def record_audio(
        self,
        duration: int = 5,
        output_file: str = "recording.wav",
        sample_rate: Optional[int] = None,
        show_timer: bool = True
    ) -> str:
        """
        Enregistre l'audio depuis le microphone.
        
        Args:
            duration (int): Durée de l'enregistrement en secondes
            output_file (str): Nom du fichier de sortie
            sample_rate (int, optional): Taux d'échantillonnage
            show_timer (bool): Affiche un compte à rebours
            
        Returns:
            str: Chemin du fichier audio enregistré
        """
        try:
            if sample_rate is None:
                sample_rate = self.sample_rate

            print(f"\nPréparation de l'enregistrement...")
            time.sleep(1)
            print("3...")
            time.sleep(1)
            print("2...")
            time.sleep(1)
            print("1...")
            time.sleep(1)
            print("\n🎤 Enregistrement en cours...")
            
            recording = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=self.channels,
                dtype=np.int16
            )
            
            if show_timer:
                for remaining in range(duration, 0, -1):
                    print(f"⏱️  Temps restant: {remaining} secondes", end='\r')
                    time.sleep(1)
                print("\n")
            else:
                sd.wait()
            
            print("✅ Enregistrement terminé!")
            
            write(output_file, sample_rate, recording)
            print(f"💾 Enregistrement sauvegardé: {output_file}\n")
            
            return output_file

        except Exception as e:
            print(f"❌ Erreur lors de l'enregistrement : {str(e)}")
            raise

    def transcribe_audio(self, file_path: str) -> dict:
        """
        Transcrit le fichier audio en texte avec détection automatique de la langue.
        
        Args:
            file_path (str): Chemin du fichier audio
            
        Returns:
            dict: Dictionnaire contenant le texte transcrit et la langue détectée
        """
        try:
            wav_path = self.convert_to_wav(file_path)
            waveform, sample_rate = self.preprocess_audio(wav_path)
            
            # Transcription avec détection automatique de la langue
            transcription = self.transcriber(
                {"array": waveform.squeeze().numpy(), "sampling_rate": sample_rate},
                return_timestamps=True
            )
            
            transcribed_text = transcription["text"].strip()
            detected_language = transcription.get("language", "Langue non détectée")
            
            print(f"\n🌐 Langue détectée: {detected_language}")
            print("📝 Transcription:", transcribed_text)
            
            return {
                "text": transcribed_text,
                "language": detected_language
            }

        except Exception as e:
            print(f"❌ Erreur lors de la transcription : {str(e)}")
            raise

    def record_and_transcribe(
        self,
        duration: int = 5,
        output_file: str = "recording.wav",
        delete_after: bool = True
    ) -> dict:
        """
        Enregistre l'audio et le transcrit directement.
        
        Args:
            duration (int): Durée de l'enregistrement en secondes
            output_file (str): Nom du fichier de sortie
            delete_after (bool): Supprime le fichier audio après transcription
            
        Returns:
            dict: Dictionnaire contenant le texte transcrit et la langue détectée
        """
        try:
            audio_file = self.record_audio(duration, output_file)
            transcription = self.transcribe_audio(audio_file)
            
            if delete_after and os.path.exists(audio_file):
                os.remove(audio_file)
                print(f"🗑️  Fichier audio supprimé: {audio_file}")
                
            return transcription

        except Exception as e:
            print(f"❌ Erreur lors de l'enregistrement et de la transcription : {str(e)}")
            raise

    def query_nemotron(
        self,
        prompt: str,
        temperature: float = 0.5,
        max_tokens: int = 1024
    ) -> None:
        """
        Interroge le modèle Nemotron 70B.
        
        Args:
            prompt (str): Le texte de la requête à envoyer au modèle
            temperature (float, optional): Paramètre de température. Par défaut 0.5
            max_tokens (int, optional): Nombre maximum de tokens. Par défaut 1024
        """
        try:
            print("\n🤖 Envoi au modèle Nemotron...")
            completion = self.client.chat.completions.create(
                model="nvidia/llama-3.1-nemotron-70b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=1,
                max_tokens=max_tokens,
                stream=True
            )
            
            print("\n💬 Réponse:")
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    print(chunk.choices[0].delta.content, end="")
            print("\n")
                    
        except Exception as e:
            print(f"❌ Erreur lors de la requête Nemotron : {str(e)}")
            raise


    def query_nemotron_text(self, prompt):
        # Exemple de logique pour générer une réponse basée sur le prompt
        # Utiliser le modèle ou une API tierce pour obtenir une réponse
        result = self.client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024
        )
        
        response_text = ""
        for chunk in result:
            if chunk.choices[0].delta.content is not None:
                response_text += chunk.choices[0].delta.content

        return response_text    

def main():
    try:
        # Remplacez par votre clé API
        api_key = "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"
        
        print("\n🌟 Démarrage de l'assistant vocal...")
        assistant = VirtualAgricultureAssistance(api_key=api_key)
        
        while True:
            try:
                # Enregistrement et transcription avec détection de langue
                transcription_result = assistant.record_and_transcribe(duration=10)
                
                if transcription_result["text"]:
                    # Traitement par le modèle
                    assistant.query_nemotron(transcription_result["text"])
                
                # Demander à l'utilisateur s'il veut continuer
                continuer = input("\n🔄 Voulez-vous poser une autre question? (o/n): ").lower()
                if continuer != 'o':
                    print("\n👋 Au revoir!")
                    break
                    
            except Exception as e:
                print(f"\n⚠️ Une erreur est survenue lors de la session : {str(e)}")
                if input("\n🔄 Voulez-vous réessayer? (o/n): ").lower() != 'o':
                    break
        
    except Exception as e:
        print(f"\n❌ Erreur générale : {str(e)}")

if __name__ == "__main__":
    main()