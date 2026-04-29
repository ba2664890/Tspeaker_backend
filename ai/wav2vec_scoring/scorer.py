"""
T.Speak — Module Wav2Vec 2.0 : Scoring Phonétique
Analyse la prononciation phonème par phonème et compare avec un locuteur natif.

Modèle : facebook/wav2vec2-large-xlsr-53 (cross-lingual)
Adapté pour mesurer la distance phonétique accent africain ↔ anglais natif.
"""

import logging
import time
from typing import Optional

import numpy as np
import torch
import torchaudio
import Levenshtein
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
)

logger = logging.getLogger("tspeak.ai")

# Phonèmes anglais (IPA simplifié)
ENGLISH_PHONEMES = [
    "AA", "AE", "AH", "AO", "AW", "AY",
    "B", "CH", "D", "DH", "EH", "ER", "EY",
    "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P",
    "R", "S", "SH", "T", "TH", "UH", "UW",
    "V", "W", "Y", "Z", "ZH",
]

# Phonèmes fréquemment problématiques pour les locuteurs africains francophones
AFRICAN_FRENCH_DIFFICULT_PHONEMES = {
    "TH": "La plupart des langues africaines n'ont pas ce son",
    "W": "Souvent confondu avec /v/ en français",
    "R": "Le /r/ roulé africain diffère de l'anglais",
    "V": "Parfois confondu avec /b/ (substrat wolof)",
    "NG": "Le /ng/ final est difficile pour les francophones",
}


class Wav2VecScorer:
    """
    Scoreur de prononciation basé sur Wav2Vec 2.0.

    Pipeline :
    1. Extraction des features audio (CNN)
    2. Contexte temporel (Transformer)
    3. Classification phonémique (CTC)
    4. Comparaison avec référence native
    5. Score par phonème + score global
    """

    MODEL_ID = "facebook/wav2vec2-large-xlsr-53"
    SAMPLE_RATE = 16000

    def __init__(self, model_id: Optional[str] = None, device: Optional[str] = None):
        self.model_id = model_id or self.MODEL_ID
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

        logger.info("Wav2VecScorer initialisé: model=%s device=%s", self.model_id, self.device)

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def processor(self):
        if self._processor is None:
            self._load_model()
        return self._processor

    def _load_model(self):
        """Charge le modèle Wav2Vec2 avec optimisations."""
        logger.info("Chargement Wav2Vec2 '%s'...", self.model_id)
        start = time.monotonic()
        self._processor = Wav2Vec2Processor.from_pretrained(self.model_id)
        model = Wav2Vec2ForCTC.from_pretrained(self.model_id)
        
        # Optimisation : Quantisation pour CPU
        if self.device == "cpu":
            logger.info("Application de la quantisation dynamique (CPU)...")
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        
        self._model = model.to(self.device)
        self._model.eval()
        logger.info("Wav2Vec2 chargé et optimisé en %.2fs", time.monotonic() - start)

    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Charge et normalise un fichier audio à 16kHz mono."""
        waveform, sample_rate = torchaudio.load(audio_path)

        # Convertir en mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resampler à 16kHz si nécessaire
        if sample_rate != self.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate, self.SAMPLE_RATE)
            waveform = resampler(waveform)

        return waveform.squeeze()

    def extract_logits(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extrait les logits phonémiques du modèle Wav2Vec2."""
        inputs = self.processor(
            waveform.numpy(),
            sampling_rate=self.SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self._model(input_values, attention_mask=attention_mask)

        return outputs.logits  # (batch, time, vocab_size)

    def decode_phonemes(self, logits: torch.Tensor) -> list[dict]:
        """
        Décode les phonèmes à partir des logits via argmax (greedy decoding).
        Retourne une liste de phonèmes avec leur probabilité.
        """
        probs = torch.softmax(logits[0], dim=-1).cpu().numpy()  # (time, vocab)
        predicted_ids = np.argmax(probs, axis=-1)

        vocab = self.processor.tokenizer.get_vocab()
        id_to_token = {v: k for k, v in vocab.items()}

        phonemes = []
        prev_id = -1
        for t_idx, token_id in enumerate(predicted_ids):
            token = id_to_token.get(token_id, "<unk>")
            if token == "<pad>":
                prev_id = -1
                continue
            if token_id == prev_id:
                continue  # Supprimer les répétitions CTC
            phonemes.append({
                "phoneme": token,
                "time_step": t_idx,
                "probability": float(probs[t_idx, token_id]),
            })
            prev_id = token_id

        return phonemes

    def score_pronunciation(
        self,
        user_audio_path: str,
        reference_text: str,
        user_text: str,
    ) -> dict:
        """
        Score complet de prononciation.

        Args:
            user_audio_path: Chemin audio utilisateur (WAV 16kHz)
            reference_text: Texte de référence attendu (question IA)
            user_text: Transcription de la réponse utilisateur (depuis Whisper)

        Returns:
            {
                "pronunciation_score": float (0-100),
                "phoneme_scores": [...],
                "difficult_phonemes": [...],
                "word_scores": {...},
                "overall_accuracy": float,
            }
        """
        start_time = time.monotonic()

        try:
            # Charger l'audio utilisateur
            waveform = self.load_audio(user_audio_path)

            # Extraire logits
            logits = self.extract_logits(waveform)

            # Décoder phonèmes
            user_phonemes = self.decode_phonemes(logits)

            # Obtenir les phonèmes de référence (via G2P)
            reference_phonemes = self._text_to_phonemes(user_text)

            # Calculer la similarité phonétique (distance d'édition normalisée)
            phoneme_accuracy = self._compute_phoneme_accuracy(
                [p["phoneme"] for p in user_phonemes],
                reference_phonemes,
            )

            # Identifier les phonèmes difficiles pour cet utilisateur
            difficult = self._identify_difficult_phonemes(user_phonemes)

            # Score global prononciation (0-100)
            pronunciation_score = round(phoneme_accuracy * 100, 2)

            # Score par mot
            word_scores = self._compute_word_scores(user_text, user_phonemes, logits)

            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            logger.info(
                "Scoring phonétique: %.0fms — score=%.1f",
                elapsed_ms, pronunciation_score,
            )

            return {
                "pronunciation_score": pronunciation_score,
                "phoneme_accuracy": float(phoneme_accuracy),
                "phoneme_count": len(user_phonemes),
                "phoneme_scores": user_phonemes[:50],  # Limiter pour la réponse API
                "difficult_phonemes": difficult,
                "word_scores": word_scores,
                "processing_ms": elapsed_ms,
            }

        except Exception as e:
            logger.error("Erreur scoring phonétique: %s", e, exc_info=True)
            # Retourner un score neutre en cas d'erreur
            return {
                "pronunciation_score": 60.0,
                "phoneme_accuracy": 0.6,
                "phoneme_count": 0,
                "phoneme_scores": [],
                "difficult_phonemes": [],
                "word_scores": {},
                "error": str(e),
            }

    def _text_to_phonemes(self, text: str) -> list[str]:
        """
        Convertit du texte en phonèmes IPA via phonemizer.
        Fallback sur un mapping simple si phonemizer non disponible.
        """
        try:
            from phonemizer import phonemize
            phonemes_str = phonemize(
                text,
                backend="espeak",
                language="en-us",
                with_stress=True,
                separator=phonemize.separator.Separator(phone=" ", word="| ", syllable=""),
            )
            return [p for p in phonemes_str.split() if p and p != "|"]
        except Exception:
            # Fallback : tokenisation simple
            return list(text.upper().replace(" ", ""))

    def _compute_phoneme_accuracy(
        self, predicted: list[str], reference: list[str]
    ) -> float:
        """
        Calcule la précision phonémique via la bibliothèque Levenshtein (optimisée).
        """
        if not reference:
            return 0.5

        # Utiliser la bibliothèque Levenshtein pour plus de rapidité
        # Convertir les listes de phonèmes en "chaînes" de caractères uniques pour Levenshtein
        # car la lib travaille sur des chaînes.
        pred_str = "".join([chr(hash(p) % 1000 + 32) for p in predicted])
        ref_str = "".join([chr(hash(p) % 1000 + 32) for p in reference])

        edit_dist = Levenshtein.distance(pred_str, ref_str)
        max_len = max(len(predicted), len(reference))
        accuracy = 1.0 - (edit_dist / max_len) if max_len > 0 else 0.5

        return float(np.clip(accuracy, 0.0, 1.0))

    def _identify_difficult_phonemes(self, phonemes: list[dict]) -> list[dict]:
        """
        Identifie les phonèmes difficiles (faible probabilité de confiance).
        Filtre les phonèmes problématiques connus pour les locuteurs africains.
        """
        difficult = []
        for p in phonemes:
            phoneme = p["phoneme"].upper()
            if p["probability"] < 0.7 and phoneme in AFRICAN_FRENCH_DIFFICULT_PHONEMES:
                difficult.append({
                    "phoneme": phoneme,
                    "probability": p["probability"],
                    "tip": AFRICAN_FRENCH_DIFFICULT_PHONEMES[phoneme],
                })
        return difficult[:10]  # Max 10 phonèmes difficiles

    def _compute_word_scores(
        self, text: str, phonemes: list[dict], logits: torch.Tensor
    ) -> dict[str, float]:
        """
        Calcule un score par mot basé sur la confiance des phonèmes.
        Retourne {mot: score_confiance} pour colorisation dans Flutter.
        """
        words = text.split()
        if not words or not phonemes:
            return {}

        # Distribuer les phonèmes entre les mots proportionnellement
        phonemes_per_word = max(1, len(phonemes) // len(words))
        word_scores = {}

        for i, word in enumerate(words):
            start_idx = i * phonemes_per_word
            end_idx = start_idx + phonemes_per_word
            word_phonemes = phonemes[start_idx:end_idx]

            if word_phonemes:
                avg_confidence = np.mean([p["probability"] for p in word_phonemes])
                word_scores[word] = round(float(avg_confidence) * 100, 1)
            else:
                word_scores[word] = 70.0  # Score neutre

        return word_scores


# ─── Singleton global ────────────────────────────────────────────────────────

_scorer_instance: Optional[Wav2VecScorer] = None


def get_scorer() -> Wav2VecScorer:
    """Retourne l'instance singleton du scoreur."""
    global _scorer_instance
    if _scorer_instance is None:
        from django.conf import settings
        model_id = getattr(settings, "WAV2VEC_MODEL", "facebook/wav2vec2-large-xlsr-53")
        _scorer_instance = Wav2VecScorer(model_id=model_id)
    return _scorer_instance
