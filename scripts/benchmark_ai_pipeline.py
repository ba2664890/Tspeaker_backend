import os
import time
import torch
import numpy as np
import soundfile as sf
import sys
import wave

# Ajouter le root du projet au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Simulation de l'environnement Django si nécessaire
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")

def create_test_audio(path, duration=3.0):
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration))
    # Un ton simple avec un peu de bruit pour simuler de la voix (très basique)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5 + np.random.normal(0, 0.05, len(t))).astype(np.float32)
    sf.write(path, audio, sr)
    return path

def benchmark_pipeline():
    print("=== T.Speak AI Pipeline Benchmark ===")
    test_audio = "bench_sample.wav"
    create_test_audio(test_audio)
    
    try:
        # 1. Benchmark Whisper
        from ai.whisper_asr.transcriber import WhisperTranscriber
        print("\n--- Whisper ASR (faster-whisper) ---")
        # Test avec différentes configurations de calcul
        for comp in ["default", "int8"]:
            print(f"Config: model=medium, compute={comp}, device={'cuda' if torch.cuda.is_available() else 'cpu'}")
            transcriber = WhisperTranscriber(model_name="medium", compute_type=comp)
            
            # Warmup
            transcriber.transcribe(test_audio)
            
            latencies = []
            for _ in range(3):
                start = time.perf_counter()
                transcriber.transcribe(test_audio)
                latencies.append(time.perf_counter() - start)
            
            avg_latency = np.mean(latencies)
            print(f"Latency moyenne: {avg_latency:.3f}s")

        # 2. Benchmark Wav2Vec Scorer
        from ai.wav2vec_scoring.scorer import Wav2VecScorer
        print("\n--- Wav2Vec Scorer (Quantized) ---")
        scorer = Wav2VecScorer()
        
        # Warmup
        scorer.score_pronunciation(test_audio, "hello world", "hello world")
        
        latencies = []
        for _ in range(3):
            start = time.perf_counter()
            scorer.score_pronunciation(test_audio, "hello world", "hello world")
            latencies.append(time.perf_counter() - start)
            
        avg_latency = np.mean(latencies)
        print(f"Latency moyenne: {avg_latency:.3f}s")
        
    except Exception as e:
        print(f"Erreur pendant le benchmark: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(test_audio):
            os.remove(test_audio)

if __name__ == "__main__":
    benchmark_pipeline()
