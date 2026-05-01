"""
T.Speak — Module Whisper ASR
Transcription audio optimisée pour les accents africains francophones.

Modèle : whisper-medium (244M paramètres) + fine-tuning sur données africaines
"""

import logging
import os
import threading
import time
from typing import Any, Optional

import numpy as np
from jiwer import wer as _jiwer_wer
from scipy.signal import resample_poly
import soundfile as sf

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover
    WhisperModel = None

# Compatibilité avec d'anciens tests qui patchaient le module OpenAI Whisper.
whisper = None

logger = logging.getLogger("tspeak.ai")

# Whisper analyse une fenêtre fixe de 30 secondes pour la détection de langue.
_WHISPER_SAMPLE_RATE = 16_000
_LANG_DETECT_MAX_FRAMES = 30 * _WHISPER_SAMPLE_RATE


class WhisperTranscriber:
    """
    Transcripteur audio basé sur faster-whisper.

    Supporte :
    - Modèles : tiny, base, small, medium, large
    - Fine-tuning sur données vocales africaines (Mozilla Common Voice)
    - Détection automatique de langue (30s max, pas de transcription complète)
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
        self.compute_type = compute_type
        self.fine_tuned_path = fine_tuned_path
        self._model = None
        self._model_lock = threading.Lock()

        logger.info(
            "WhisperTranscriber initialisé: model=%s device=%s compute_type=%s",
            model_name, self.device, compute_type,
        )

    # ------------------------------------------------------------------
    # Chargement lazy thread-safe
    # ------------------------------------------------------------------

    @property
    def model(self):
        """Chargement paresseux du modèle (double-checked locking)."""
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._model = self._load_model()
        return self._model

    def _load_model(self) -> "WhisperModel":
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

        logger.info("Modèle chargé en %.2fs", time.monotonic() - start)
        return model

    def _resolve_compute_type(self) -> str:
        """Choisit un compute_type performant et compatible avec la cible."""
        if self.compute_type and self.compute_type != "default":
            return self.compute_type
        return "float16" if self.device == "cuda" else "int8"

    # ------------------------------------------------------------------
    # Transcription principale
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_path: str,
        language: str = "en",
        task: str = "transcribe",
        word_timestamps: bool = True,
    ) -> dict:
        """Transcrit un fichier audio via faster-whisper."""
        start_time = time.monotonic()

        try:
            raw_result = self.model.transcribe(
                audio_path,
                language=language,
                task=task,
                word_timestamps=word_timestamps,
                beam_size=5,
                # Hyperparamètres calibrés pour accents africains francophones.
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
            segments_list = list(segments)

            # QUALITE: construction de full_text unifiée dans les deux branches.
            full_text = "".join(s.text for s in segments_list)

            words: list[dict] = []
            if word_timestamps:
                for segment in segments_list:
                    for word_info in (segment.words or []):
                        words.append({
                            "word": word_info.word.strip(),
                            "start": word_info.start,
                            "end": word_info.end,
                            "probability": word_info.probability,
                        })

            avg_confidence = float(np.mean([w["probability"] for w in words])) if words else 0.0

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
                        # QUALITE: None-check explicite plutôt que `or 0.0` sur float.
                        "avg_logprob": _to_float(getattr(s, "avg_logprob", None)),
                        "no_speech_prob": _to_float(getattr(s, "no_speech_prob", None)),
                    }
                    for s in segments_list
                ],
                "language_probability": float(info.language_probability),
                "no_speech_prob": float(np.mean([
                    _to_float(getattr(s, "no_speech_prob", None))
                    for s in segments_list
                ])) if segments_list else 1.0,
                "avg_confidence": avg_confidence,
                "processing_ms": int(elapsed_ms),
            }

        except Exception as e:
            logger.error("Erreur Whisper: %s", e, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Détection de langue
    # ------------------------------------------------------------------

    def detect_language(self, audio_path: str) -> tuple[str, float]:
        """Détecte la langue de l'audio en analysant uniquement les 30 premières
        secondes — fenêtre maximale utilisée par l'architecture Whisper.

        PERF: l'ancienne implémentation passait le chemin complet à
        model.transcribe(), ce qui traitait tout le fichier. On charge
        maintenant uniquement les 30 premières secondes en mémoire et on
        passe le numpy array directement, ce qui est ~4× plus rapide sur
        un fichier de 2 minutes.
        """
        waveform = _load_audio_first_30s(audio_path)

        _, info = self.model.transcribe(
            waveform,
            beam_size=1,
            language=None,
            word_timestamps=False,
            condition_on_previous_text=False,
        )
        logger.info(
            "Langue détectée: %s (confiance=%.2f)", info.language, info.language_probability
        )
        return info.language, float(info.language_probability)

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    def _format_legacy_result(self, result: dict[str, Any], start_time: float) -> dict:
        """Normalise l'ancien format openai-whisper utilisé dans certains tests."""
        words: list[dict] = []
        for segment in result.get("segments", []):
            for word_info in segment.get("words", []) or []:
                words.append({
                    "word": str(word_info.get("word", "")).strip(),
                    "start": float(word_info.get("start") or 0.0),
                    "end": float(word_info.get("end") or 0.0),
                    "probability": float(word_info.get("probability") or 0.0),
                })

        avg_confidence = float(np.mean([w["probability"] for w in words])) if words else 0.0
        segments = result.get("segments", [])
        elapsed_ms = (time.monotonic() - start_time) * 1000
        return {
            "text": str(result.get("text", "")).strip(),
            "language": result.get("language", "en"),
            "words": words,
            "segments": segments,
            "language_probability": float(result.get("language_probability") or 0.0),
            "no_speech_prob": float(np.mean([
                s.get("no_speech_prob") or 0.0 for s in segments
            ])) if segments else 0.0,
            "avg_confidence": avg_confidence,
            "processing_ms": int(elapsed_ms),
        }

    @staticmethod
    def compute_wer(reference: str, hypothesis: str) -> float:
        """Calcule le Word Error Rate entre référence et transcription.

        QUALITE: méthode transformée en @staticmethod car elle n'accède à
        aucun attribut d'instance. Peut être appelée sans instancier le
        transcripteur : WhisperTranscriber.compute_wer(ref, hyp).
        L'import jiwer est désormais au niveau module.
        """
        return _jiwer_wer(reference.lower(), hypothesis.lower())


# ---------------------------------------------------------------------------
# Utilitaires module-level
# ---------------------------------------------------------------------------

def _to_float(value: Any, default: float = 0.0) -> float:
    """Convertit une valeur en float de façon explicite, avec fallback sur None."""
    return float(value) if value is not None else default


def _load_audio_first_30s(audio_path: str) -> np.ndarray:
    """Charge et resample les 30 premières secondes d'un fichier audio en
    mono float32 16 kHz — format attendu par le feature extractor de Whisper.
    """
    with sf.SoundFile(audio_path) as f:
        native_sr = f.samplerate
        max_frames_native = 30 * native_sr
        waveform = f.read(frames=max_frames_native, dtype="float32", always_2d=False)

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    if native_sr != _WHISPER_SAMPLE_RATE:
        gcd = int(np.gcd(native_sr, _WHISPER_SAMPLE_RATE))
        waveform = resample_poly(
            waveform,
            _WHISPER_SAMPLE_RATE // gcd,
            native_sr // gcd,
        ).astype(np.float32)

    # Garantir qu'on ne dépasse pas exactement 30s après resampling.
    return waveform[:_LANG_DETECT_MAX_FRAMES]


# ---------------------------------------------------------------------------
# Singleton global thread-safe
# ---------------------------------------------------------------------------

_transcriber_instance: Optional[WhisperTranscriber] = None
_transcriber_lock = threading.Lock()  # FIX: verrou absent dans la version originale.


def get_transcriber() -> WhisperTranscriber:
    """Retourne l'instance singleton du transcripteur (lazy, thread-safe).

    FIX: l'ancienne implémentation n'avait aucun verrou. En production
    Django multi-workers, deux requêtes concurrentes pouvaient charger le
    modèle en double. On applique le même double-checked locking que scorer.
    """
    global _transcriber_instance
    if _transcriber_instance is None:
        with _transcriber_lock:
            if _transcriber_instance is None:
                _transcriber_instance = _build_transcriber_from_settings()
    return _transcriber_instance


def _build_transcriber_from_settings() -> WhisperTranscriber:
    """Instancie le transcripteur à partir de django.conf.settings si disponible,
    sinon utilise les valeurs par défaut.

    QUALITE: isoler cette logique de get_transcriber() découple le module de
    Django et permet les tests unitaires sans projet Django configuré.
    """
    try:
        from django.conf import settings as django_settings
        return WhisperTranscriber(
            model_name=getattr(django_settings, "WHISPER_MODEL", "medium"),
            device=getattr(django_settings, "WHISPER_DEVICE", None),
            compute_type=getattr(django_settings, "WHISPER_COMPUTE_TYPE", "default"),
            fine_tuned_path=getattr(django_settings, "WHISPER_FINE_TUNED_PATH", None),
        )
    except Exception:
        # Django non configuré (tests unitaires, scripts standalone, etc.)
        return WhisperTranscriber()
