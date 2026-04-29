import os
import sys
import torch
import numpy as np
from io import BytesIO
import wave
import struct

# Ajouter le chemin du projet au PYTHONPATH
sys.path.append(os.getcwd())

def create_fake_audio(path):
    """Crée un fichier WAV factice de 2 secondes avec un ton pur."""
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    # Un ton simple à 440Hz
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float16) # pcm_s16le
    
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())
    return path

def test_pipeline():
    print("🚀 Démarrage du test du pipeline IA T.Speak...")
    
    test_audio = "test_sample.wav"
    create_fake_audio(test_audio)
    print(f"✅ Fichier audio de test généré : {test_audio}")

    try:
        # 1. Test Whisper
        from ai.whisper_asr.transcriber import WhisperTranscriber
        import soundfile as sf
        
        print("📥 Chargement de Whisper (modèle 'tiny' pour le test)...")
        transcriber = WhisperTranscriber(model_name="tiny", device="cpu")
        
        print("🎙️ Transcription en cours (via NumPy array pour éviter ffmpeg)...")
        # Charger l'audio manuellement pour éviter que Whisper appelle ffmpeg
        audio_data, samplerate = sf.read(test_audio)
        audio_data = audio_data.astype(np.float32)
        result = transcriber.transcribe(audio_data, language="en")
        print(f"✅ Transcription réussie : '{result['text']}'")
        print(f"📊 Confiance : {result['avg_confidence']:.2f}")

        # 2. Test Wav2Vec Scoring
        from ai.wav2vec_scoring.scorer import Wav2VecScorer
        print("📥 Chargement de Wav2VecScorer...")
        scorer = Wav2VecScorer(device="cpu")
        
        print("📏 Scoring phonétique en cours...")
        score_result = scorer.score_pronunciation(
            test_audio, 
            reference_text="hello world", 
            user_text=result["text"]
        )
        print(f"✅ Score Prononciation : {score_result['pronunciation_score']}/100")
        print(f"🔍 Phonèmes détectés : {len(score_result['phoneme_scores'])}")

        print("\n✨ Test du pipeline IA terminé avec succès !")

    except Exception as e:
        print(f"\n❌ Erreur pendant le test : {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(test_audio):
            os.remove(test_audio)

if __name__ == "__main__":
    test_pipeline()
