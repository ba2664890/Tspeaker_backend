"""
T.Speak - Wav2Vec 2.0 pronunciation scoring.

Le scoreur utilise un vrai checkpoint Wav2Vec2ForCTC anglais pour mesurer la
qualite acoustique, puis combine cette evidence avec la transcription Whisper et
le texte attendu. Le resultat reste exploitable meme sans phonemizer systeme.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from jiwer import wer

try:
    import Levenshtein
except ImportError:  # pragma: no cover - dependance declaree, fallback de demarrage.
    Levenshtein = None

try:
    import torch
except ImportError:  # pragma: no cover - permet a Django de demarrer avant install IA.
    torch = None

logger = logging.getLogger("tspeak.ai")


AFRICAN_FRENCH_DIFFICULT_PHONEMES = {
    "TH": "Place la langue entre les dents; evite de remplacer par /s/ ou /z/.",
    "DH": "Garde la langue entre les dents et fais vibrer la voix.",
    "R": "L'anglais demande un /r/ retroflechi, moins roule que le /r/ local.",
    "W": "Arrondis les levres; evite de le transformer en /v/.",
    "V": "Mets les dents du haut sur la levre du bas; evite /b/.",
    "NG": "Garde le son dans le nez en fin de mot, sans ajouter /g/ fort.",
    "H": "Expire clairement au debut du mot; ne le rends pas muet.",
}


@dataclass(frozen=True)
class TokenSpan:
    token: str
    start_sec: float
    end_sec: float
    probability: float


class Wav2VecScorer:
    """
    Scoreur de prononciation rapide et complet.

    Pipeline:
    1. Charge l'audio en mono 16 kHz.
    2. Passe l'audio dans un modele Wav2Vec2 CTC anglais.
    3. Decode les tokens avec timestamps et confiances.
    4. Compare reference, transcription Whisper et sortie CTC.
    5. Retourne score global, details par token/mot et conseils ciblés.
    """

    MODEL_ID = "facebook/wav2vec2-base-960h"
    SAMPLE_RATE = 16000
    MAX_AUDIO_SECONDS = 120

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        max_audio_seconds: int = MAX_AUDIO_SECONDS,
    ):
        self.model_id = model_id or self.MODEL_ID
        self.device = device or (
            "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        )
        self.max_audio_seconds = max_audio_seconds
        self._model = None
        self._processor = None
        self._blank_token_id: Optional[int] = None
        self._model_lock = threading.Lock()

        logger.info("Wav2VecScorer initialise: model=%s device=%s", self.model_id, self.device)

    @property
    def model(self):
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._load_model()
        return self._model

    @property
    def processor(self):
        if self._processor is None:
            with self._model_lock:
                if self._processor is None:
                    self._load_model()
        return self._processor

    def _load_model(self):
        """Charge le modele Wav2Vec2ForCTC et applique les optimisations CPU/GPU."""
        if torch is None:
            raise RuntimeError(
                "torch est requis pour le scoring Wav2Vec. Installez les dependances backend."
            )

        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        logger.info("Chargement Wav2Vec2 CTC '%s'...", self.model_id)
        start = time.monotonic()

        processor = Wav2Vec2Processor.from_pretrained(self.model_id)
        model = Wav2Vec2ForCTC.from_pretrained(self.model_id)

        if self.device == "cpu":
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        else:
            model = model.to(self.device)

        model.eval()
        self._processor = processor
        self._model = model
        self._blank_token_id = getattr(processor.tokenizer, "pad_token_id", None)
        logger.info("Wav2Vec2 charge en %.2fs", time.monotonic() - start)

    def load_audio(self, audio_path: str) -> np.ndarray:
        """Charge et normalise l'audio en mono float32 16 kHz."""
        info = sf.info(audio_path)
        sample_rate = int(info.samplerate)
        waveform, sample_rate = sf.read(
            audio_path,
            dtype="float32",
            always_2d=False,
            frames=self.max_audio_seconds * sample_rate,
        )
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sample_rate != self.SAMPLE_RATE:
            gcd = np.gcd(sample_rate, self.SAMPLE_RATE)
            waveform = resample_poly(
                waveform,
                self.SAMPLE_RATE // gcd,
                sample_rate // gcd,
            ).astype(np.float32)
        if waveform.size == 0:
            raise ValueError("Audio vide ou illisible.")

        waveform = np.asarray(waveform, dtype=np.float32)
        peak = float(np.max(np.abs(waveform)))
        if peak > 0:
            waveform = waveform / peak * 0.95
        return waveform

    def extract_logits(self, waveform: np.ndarray) -> torch.Tensor:
        """Extrait les logits CTC du modele."""
        inputs = self.processor(
            waveform,
            sampling_rate=self.SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.inference_mode():
            outputs = self.model(input_values, attention_mask=attention_mask)
        return outputs.logits

    def decode_tokens(self, logits: torch.Tensor, audio_duration_sec: float) -> tuple[str, list[TokenSpan]]:
        """Decode CTC greedy avec collapse des repetitions et timestamps approximes."""
        probs = torch.softmax(logits[0], dim=-1).detach().cpu().numpy()
        predicted_ids = np.argmax(probs, axis=-1)
        id_to_token = {v: k for k, v in self.processor.tokenizer.get_vocab().items()}
        time_step_sec = audio_duration_sec / max(1, len(predicted_ids))

        collapsed: list[TokenSpan] = []
        transcript_parts: list[str] = []
        prev_id: Optional[int] = None

        for idx, token_id in enumerate(predicted_ids):
            if token_id == self._blank_token_id:
                prev_id = None
                continue
            if prev_id == token_id:
                continue

            token = id_to_token.get(int(token_id), "")
            if not token or token.startswith("<"):
                prev_id = token_id
                continue

            normalized_token = " " if token == "|" else token.lower()
            transcript_parts.append(normalized_token)
            collapsed.append(TokenSpan(
                token=normalized_token.strip() or "|",
                start_sec=idx * time_step_sec,
                end_sec=(idx + 1) * time_step_sec,
                probability=float(probs[idx, token_id]),
            ))
            prev_id = token_id

        ctc_text = _normalize_text("".join(transcript_parts))
        return ctc_text, collapsed

    def score_pronunciation(
        self,
        user_audio_path: str,
        reference_text: str,
        user_text: str,
    ) -> dict:
        start_time = time.monotonic()

        try:
            waveform = self.load_audio(user_audio_path)
            duration_sec = len(waveform) / self.SAMPLE_RATE
            logits = self.extract_logits(waveform)
            ctc_text, token_spans = self.decode_tokens(logits, duration_sec)

            normalized_reference = _normalize_text(reference_text)
            normalized_user = _normalize_text(user_text) or ctc_text

            reference_match = _similarity(normalized_reference, normalized_user)
            ctc_match = _similarity(normalized_user, ctc_text)
            phoneme_accuracy = self._phoneme_similarity(normalized_reference, normalized_user)
            acoustic_confidence = self._acoustic_confidence(token_spans)

            pronunciation_score = _weighted_score(
                acoustic_confidence=acoustic_confidence,
                reference_match=reference_match,
                ctc_match=ctc_match,
                phoneme_accuracy=phoneme_accuracy,
            )
            word_scores = self._compute_word_scores(normalized_user, token_spans, pronunciation_score)
            difficult = self._identify_difficult_phonemes(normalized_reference, normalized_user, token_spans)

            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            logger.info(
                "Scoring Wav2Vec: %dms score=%.1f acoustic=%.2f ref=%.2f ctc=%.2f",
                elapsed_ms,
                pronunciation_score,
                acoustic_confidence,
                reference_match,
                ctc_match,
            )

            return {
                "pronunciation_score": pronunciation_score,
                "phoneme_accuracy": round(float(phoneme_accuracy), 4),
                "acoustic_confidence": round(float(acoustic_confidence), 4),
                "reference_match": round(float(reference_match), 4),
                "ctc_match": round(float(ctc_match), 4),
                "ctc_transcription": ctc_text,
                "phoneme_count": len(token_spans),
                "phoneme_scores": [
                    {
                        "phoneme": span.token.upper(),
                        "start": round(span.start_sec, 3),
                        "end": round(span.end_sec, 3),
                        "probability": round(span.probability, 4),
                    }
                    for span in token_spans[:80]
                ],
                "difficult_phonemes": difficult,
                "word_scores": word_scores,
                "processing_ms": elapsed_ms,
            }

        except Exception as exc:
            logger.error("Erreur scoring Wav2Vec: %s", exc, exc_info=True)
            return self._fallback_score(reference_text, user_text, str(exc), start_time)

    def _phoneme_similarity(self, reference_text: str, user_text: str) -> float:
        reference_units = _text_to_pronunciation_units(reference_text)
        user_units = _text_to_pronunciation_units(user_text)
        return _sequence_similarity(reference_units, user_units)

    def _acoustic_confidence(self, token_spans: list[TokenSpan]) -> float:
        if not token_spans:
            return 0.35
        probabilities = [span.probability for span in token_spans if span.token != "|"]
        if not probabilities:
            return 0.35
        # On compresse legerement les extremes: les CTC sont souvent trop confiants.
        confidence = float(np.mean(probabilities))
        return float(np.clip((confidence - 0.15) / 0.8, 0.0, 1.0))

    def _identify_difficult_phonemes(
        self,
        reference_text: str,
        user_text: str,
        token_spans: list[TokenSpan],
    ) -> list[dict]:
        hints: list[dict] = []
        reference_upper = reference_text.upper()
        user_upper = user_text.upper()
        low_confidence = {
            span.token.upper()
            for span in token_spans
            if span.token != "|" and span.probability < 0.55
        }

        candidates = {
            "TH": ("TH" in reference_upper and "TH" not in user_upper) or {"T", "S", "Z"} & low_confidence,
            "DH": ("THE" in reference_upper or "THAT" in reference_upper) and {"D", "Z"} & low_confidence,
            "R": "R" in reference_upper and "R" in low_confidence,
            "W": "W" in reference_upper and "W" in low_confidence,
            "V": "V" in reference_upper and "V" in low_confidence,
            "NG": "NG" in reference_upper and "NG" not in user_upper,
            "H": reference_upper.startswith("H") and (not user_upper.startswith("H") or "H" in low_confidence),
        }
        for phoneme, is_difficult in candidates.items():
            if is_difficult:
                hints.append({
                    "phoneme": phoneme,
                    "probability": round(_average_token_probability(phoneme, token_spans), 4),
                    "tip": AFRICAN_FRENCH_DIFFICULT_PHONEMES[phoneme],
                })
        return hints[:10]

    def _compute_word_scores(
        self,
        text: str,
        token_spans: list[TokenSpan],
        fallback_score: float,
    ) -> dict[str, float]:
        words = text.split()
        if not words:
            return {}
        if not token_spans:
            return {word: round(fallback_score, 1) for word in words}

        letter_spans = [span for span in token_spans if re.match(r"[a-z]", span.token)]
        scores: dict[str, float] = {}
        cursor = 0
        for word in words:
            letters_needed = len(re.sub(r"[^a-z]", "", word.lower()))
            chunk = letter_spans[cursor:cursor + max(1, letters_needed)]
            cursor += letters_needed
            if chunk:
                scores[word] = round(float(np.mean([s.probability for s in chunk])) * 100, 1)
            else:
                scores[word] = round(fallback_score, 1)
        return scores

    def _fallback_score(
        self,
        reference_text: str,
        user_text: str,
        error: str,
        start_time: float,
    ) -> dict:
        normalized_reference = _normalize_text(reference_text)
        normalized_user = _normalize_text(user_text)
        text_similarity = _similarity(normalized_reference, normalized_user)
        score = round(float(np.clip(45 + text_similarity * 40, 35, 85)), 2)
        return {
            "pronunciation_score": score,
            "phoneme_accuracy": round(text_similarity, 4),
            "acoustic_confidence": 0.0,
            "reference_match": round(text_similarity, 4),
            "ctc_match": 0.0,
            "ctc_transcription": "",
            "phoneme_count": 0,
            "phoneme_scores": [],
            "difficult_phonemes": [],
            "word_scores": {
                word: score for word in normalized_user.split()
            },
            "processing_ms": int((time.monotonic() - start_time) * 1000),
            "error": error,
        }


def _normalize_text(text: str) -> str:
    text = text or ""
    text = text.lower().replace("'", "")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _similarity(reference: str, hypothesis: str) -> float:
    if not reference and not hypothesis:
        return 1.0
    if not reference or not hypothesis:
        return 0.0

    word_accuracy = 1.0 - min(1.0, wer(reference, hypothesis))
    if Levenshtein is not None:
        char_distance = Levenshtein.distance(reference, hypothesis)
    else:
        char_distance = _edit_distance(list(reference), list(hypothesis))
    char_accuracy = 1.0 - char_distance / max(len(reference), len(hypothesis), 1)
    return float(np.clip((word_accuracy * 0.65) + (char_accuracy * 0.35), 0.0, 1.0))


def _sequence_similarity(reference: list[str], hypothesis: list[str]) -> float:
    if not reference and not hypothesis:
        return 1.0
    if not reference or not hypothesis:
        return 0.0
    distance = _edit_distance(reference, hypothesis)
    return float(np.clip(1.0 - distance / max(len(reference), len(hypothesis), 1), 0.0, 1.0))


def _edit_distance(a: list[str], b: list[str]) -> int:
    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (0 if item_a == item_b else 1),
            ))
        previous = current
    return previous[-1]


def _text_to_pronunciation_units(text: str) -> list[str]:
    try:
        from phonemizer import phonemize
        from phonemizer.separator import Separator

        phonemes = phonemize(
            text,
            backend="espeak",
            language="en-us",
            strip=True,
            preserve_punctuation=False,
            with_stress=False,
            separator=Separator(phone=" ", word=" | ", syllable=""),
        )
        units = [unit for unit in phonemes.split() if unit != "|"]
        if units:
            return units
    except Exception:
        pass

    normalized = _normalize_text(text)
    # Fallback graphemique: moins fin que l'IPA, mais stable et sans dependance systeme.
    return [char for char in normalized.replace(" ", "|")]


def _weighted_score(
    acoustic_confidence: float,
    reference_match: float,
    ctc_match: float,
    phoneme_accuracy: float,
) -> float:
    score = (
        acoustic_confidence * 0.35
        + reference_match * 0.30
        + phoneme_accuracy * 0.25
        + ctc_match * 0.10
    ) * 100
    return round(float(np.clip(score, 0.0, 100.0)), 2)


def _average_token_probability(token: str, token_spans: list[TokenSpan]) -> float:
    values = [span.probability for span in token_spans if span.token.upper() in token]
    return float(np.mean(values)) if values else 0.0


_scorer_instance: Optional[Wav2VecScorer] = None
_scorer_lock = threading.Lock()


def get_scorer() -> Wav2VecScorer:
    """Retourne l'instance singleton du scoreur."""
    global _scorer_instance
    if _scorer_instance is None:
        with _scorer_lock:
            if _scorer_instance is None:
                from django.conf import settings

                _scorer_instance = Wav2VecScorer(
                    model_id=getattr(settings, "WAV2VEC_MODEL", Wav2VecScorer.MODEL_ID),
                    device=getattr(settings, "WAV2VEC_DEVICE", None),
                    max_audio_seconds=getattr(
                        settings,
                        "WAV2VEC_MAX_AUDIO_SECONDS",
                        Wav2VecScorer.MAX_AUDIO_SECONDS,
                    ),
                )
    return _scorer_instance
