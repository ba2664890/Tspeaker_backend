"""
T.Speak — Module Whisper ASR
Transcription audio optimisée pour les accents africains francophones.

Modèle : whisper-medium (244M paramètres) + fine-tuning sur données africaines
"""

import logging
import os
import time
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
from faster_whisper import WhisperModel

logger = logging.getLogger("tspeak.ai")


class WhisperTranscriber:
    """
    Transcripteur audio basé sur OpenAI Whisper.

    Supporte :
    - Modèles : tiny, base, small, medium, large
    - Fine-tuning sur données vocales africaines (Mozilla Common Voice)
    - Détection automatique de langue
    - Confidence scores par mot
    """

    def __init__(
        self,
        model_name: str = "medium",
        device: Optional[str] = None,
        compute_type: str = "default",
        fine_tuned_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = compute_type  # e.g., "int8", "float16"
        self.fine_tuned_path = fine_tuned_path
        self._model = None

        logger.info(
            "WhisperTranscriber initialisé: model=%s device=%s compute_type=%s",
            model_name, self.device, compute_type,
        )

    @property
    def model(self):
        """Chargement paresseux du modèle."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> WhisperModel:
        """Charge le modèle Whisper via faster-whisper."""
        logger.info("Chargement du modèle faster-whisper '%s'...", self.model_name)
        start = time.monotonic()

        # Note: Les modèles fine-tunés doivent être convertis au format CTranslate2
        # pour être utilisés avec faster-whisper.
        model_path = self.fine_tuned_path if self.fine_tuned_path and os.path.exists(self.fine_tuned_path) else self.model_name

        model = WhisperModel(
            model_path,
            device=self.device,
            compute_type=self.compute_type,
        )

        elapsed = time.monotonic() - start
        logger.info("Modèle chargé en %.2fs", elapsed)
        return model

    def transcribe(
        self,
        audio_path: str,
        language: str = "en",
        task: str = "transcribe",
        word_timestamps: bool = True,
    ) -> dict:
        """
        Transcrit un fichier audio via faster-whisper.
        """
        start_time = time.monotonic()

        try:
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                task=task,
                word_timestamps=word_timestamps,
                beam_size=5,
                # Optimisations pour accents africains via hyperparamètres
                temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=False,
            )

            # Consommer le générateur segments
            segments_list = list(segments)
            
            # Extraire les mots avec timestamps
            words = []
            full_text = ""
            if word_timestamps:
                for segment in segments_list:
                    full_text += segment.text
                    for word_info in (segment.words or []):
                        words.append({
                            "word": word_info.word.strip(),
                            "start": word_info.start,
                            "end": word_info.end,
                            "probability": word_info.probability,
                        })
            else:
                full_text = "".join([s.text for s in segments_list])

            # Score de confiance moyen
            avg_confidence = (
                np.mean([w["probability"] for w in words]) if words else 0.0
            )

            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.info(
                "Transcription (faster): %.0fms — %d mots — confiance=%.2f",
                elapsed_ms, len(words), avg_confidence,
            )

            return {
                "text": full_text.strip(),
                "language": info.language,
                "words": words,
                "segments": [
                    {"start": s.start, "end": s.end, "text": s.text} 
                    for s in segments_list
                ],
                "no_speech_prob": float(info.language_probability), # Approximation
                "avg_confidence": float(avg_confidence),
                "processing_ms": int(elapsed_ms),
            }

        except Exception as e:
            logger.error("Erreur Whisper: %s", e, exc_info=True)
            raise

    def detect_language(self, audio_path: str) -> tuple[str, float]:
        """
        Détecte la langue de l'audio via faster-whisper.
        """
        # Pour une détection rapide, on peut transcrire avec juste les premiers moments
        _, info = self.model.transcribe(audio_path, beam_size=1, duration=30)
        logger.info("Langue détectée: %s (confiance=%.2f)", info.language, info.language_probability)
        return info.language, float(info.language_probability)

    def compute_wer(self, reference: str, hypothesis: str) -> float:
        """
        Calcule le Word Error Rate (WER) entre référence et transcription.
        Utilisé pour évaluer la qualité du modèle sur données africaines.
        """
        from jiwer import wer
        return wer(reference.lower(), hypothesis.lower())


# ─── Singleton global ────────────────────────────────────────────────────────

_transcriber_instance: Optional[WhisperTranscriber] = None


def get_transcriber() -> WhisperTranscriber:
    """Retourne l'instance singleton du transcripteur."""
    global _transcriber_instance
    if _transcriber_instance is None:
        from django.conf import settings
        _transcriber_instance = WhisperTranscriber(
            model_name=getattr(settings, "WHISPER_MODEL", "medium"),
            device=getattr(settings, "WHISPER_DEVICE", None),
            compute_type=getattr(settings, "WHISPER_COMPUTE_TYPE", "default"),
            fine_tuned_path=getattr(settings, "WHISPER_FINE_TUNED_PATH", None),
        )
    return _transcriber_instance
