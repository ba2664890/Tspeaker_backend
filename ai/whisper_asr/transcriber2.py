"""
T.Speak — Module Whisper ASR
Transcription audio optimisée pour les accents africains francophones.

Modèle : whisper-medium (244M paramètres) + fine-tuning sur données africaines
"""

import logging
import os
import threading
import time
from functools import lru_cache
from typing import Any, Optional

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - dependance declaree, fallback de demarrage.
    torch = None

try:
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover - erreur explicite au chargement du modele.
    WhisperModel = None

# Compatibilité avec d'anciens tests qui patchaient le module OpenAI Whisper.
whisper = None

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
        self.device = device or (
            "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        )
        self.compute_type = compute_type  # e.g., "int8", "float16"
        self.fine_tuned_path = fine_tuned_path
        self._model = None
        self._model_lock = threading.Lock()

        logger.info(
            "WhisperTranscriber initialisé: model=%s device=%s compute_type=%s",
            model_name, self.device, compute_type,
        )

    @property
    def model(self):
        """Chargement paresseux du modèle."""
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._model = self._load_model()
        return self._model

    def _load_model(self) -> WhisperModel:
        """Charge le modèle Whisper via faster-whisper."""
        if WhisperModel is None:
            raise RuntimeError(
                "faster-whisper est requis pour l'ASR. Installez les dependances backend."
            )

        logger.info("Chargement du modèle faster-whisper '%s'...", self.model_name)
        start = time.monotonic()

        model_path = (
            self.fine_tuned_path
            if self.fine_tuned_path and os.path.exists(self.fine_tuned_path)
            else self.model_name
        )

        model = WhisperModel(
            model_path,
            device=self.device,
            compute_type=self._resolve_compute_type(),
        )

        elapsed = time.monotonic() - start
        logger.info("Modèle chargé en %.2fs", elapsed)
        return model

    def _resolve_compute_type(self) -> str:
        """Choisit un compute_type performant et compatible avec la cible."""
        if self.compute_type and self.compute_type != "default":
            return self.compute_type
        if self.device == "cuda":
            return "float16"
        return "int8"

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
            raw_result = self.model.transcribe(
                audio_path,
                language=language,
                task=task,
                word_timestamps=word_timestamps,
                beam_size=5,
                # Optimisations pour accents africains via hyperparamètres
                temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                condition_on_previous_text=False,
            )

            if isinstance(raw_result, dict):
                return self._format_legacy_result(raw_result, start_time)

            segments, info = raw_result
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
                    {
                        "start": float(s.start),
                        "end": float(s.end),
                        "text": s.text.strip(),
                        "avg_logprob": float(getattr(s, "avg_logprob", 0.0) or 0.0),
                        "no_speech_prob": float(getattr(s, "no_speech_prob", 0.0) or 0.0),
                    }
                    for s in segments_list
                ],
                "language_probability": float(info.language_probability),
                "no_speech_prob": float(np.mean([
                    getattr(s, "no_speech_prob", 0.0) or 0.0
                    for s in segments_list
                ])) if segments_list else 1.0,
                "avg_confidence": float(avg_confidence),
                "processing_ms": int(elapsed_ms),
            }

        except Exception as e:
            logger.error("Erreur Whisper: %s", e, exc_info=True)
            raise

    def _format_legacy_result(self, result: dict[str, Any], start_time: float) -> dict:
        """Normalise l'ancien format openai-whisper utilisé dans certains tests."""
        words = []
        for segment in result.get("segments", []):
            for word_info in segment.get("words", []) or []:
                words.append({
                    "word": str(word_info.get("word", "")).strip(),
                    "start": float(word_info.get("start", 0.0) or 0.0),
                    "end": float(word_info.get("end", 0.0) or 0.0),
                    "probability": float(word_info.get("probability", 0.0) or 0.0),
                })

        avg_confidence = np.mean([w["probability"] for w in words]) if words else 0.0
        elapsed_ms = (time.monotonic() - start_time) * 1000
        return {
            "text": str(result.get("text", "")).strip(),
            "language": result.get("language", "en"),
            "words": words,
            "segments": result.get("segments", []),
            "language_probability": float(result.get("language_probability", 0.0) or 0.0),
            "no_speech_prob": float(np.mean([
                s.get("no_speech_prob", 0.0) or 0.0
                for s in result.get("segments", [])
            ])) if result.get("segments") else 0.0,
            "avg_confidence": float(avg_confidence),
            "processing_ms": int(elapsed_ms),
        }

    def detect_language(self, audio_path: str) -> tuple[str, float]:
        """
        Détecte la langue de l'audio via faster-whisper.
        """
        _, info = self.model.transcribe(
            audio_path,
            beam_size=1,
            language=None,
            word_timestamps=False,
            condition_on_previous_text=False,
        )
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
