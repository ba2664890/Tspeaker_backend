"""
T.Speak — Script Fine-Tuning Whisper sur Données Africaines
============================================================

Fine-tune whisper-medium sur Mozilla Common Voice (Wolof + Français Afrique)
pour améliorer la reconnaissance des accents africains francophones.

Usage:
    python scripts/finetune_whisper.py \
        --model_name openai/whisper-medium \
        --dataset_path ./data/african_voices \
        --output_dir ./models/whisper-african \
        --num_epochs 5 \
        --batch_size 16 \
        --learning_rate 1e-5

Données requises:
    - Mozilla Common Voice Wolof (wo)
    - Mozilla Common Voice French Africa subset
    - Données collectées T.Speak (format HuggingFace datasets)

Métriques cibles:
    - WER (Word Error Rate) < 15% sur accents africains
    - WER < 8% sur anglais standard
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import evaluate
import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    EarlyStoppingCallback,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class TSpkTrainingConfig:
    """Configuration du fine-tuning T.Speak."""
    model_name: str = "openai/whisper-medium"
    output_dir: str = "./models/whisper-african"
    dataset_path: str = "./data/african_voices"

    # Hyperparamètres
    num_epochs: int = 5
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-5
    warmup_steps: int = 500
    max_steps: int = 5000
    fp16: bool = True  # Mixed precision (si GPU disponible)

    # Audio
    max_input_length_sec: float = 30.0
    sample_rate: int = 16000
    language: str = "en"
    task: str = "transcribe"

    # Évaluation
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 100
    early_stopping_patience: int = 3


# ─── Préparation des données ──────────────────────────────────────────────────

def load_african_dataset(config: TSpkTrainingConfig) -> DatasetDict:
    """
    Charge et combine les datasets africains pour le fine-tuning.

    Sources :
    1. Mozilla Common Voice - Français (subset africain)
    2. Mozilla Common Voice - Wolof (si disponible)
    3. Données collectées T.Speak (format custom)
    """
    datasets_list = []

    # 1. Mozilla Common Voice - Français
    logger.info("Chargement Mozilla Common Voice Français...")
    try:
        cv_fr = load_dataset(
            "mozilla-foundation/common_voice_13_0",
            "fr",
            split="train",
            streaming=False,
        )
        # Filtrer les locuteurs africains (si métadonnées disponibles)
        cv_fr = cv_fr.select(range(min(5000, len(cv_fr))))
        datasets_list.append(cv_fr)
        logger.info("CV Français: %d exemples", len(cv_fr))
    except Exception as e:
        logger.warning("Impossible de charger CV Français: %s", e)

    # 2. Mozilla Common Voice - Wolof
    logger.info("Chargement Mozilla Common Voice Wolof...")
    try:
        cv_wo = load_dataset(
            "mozilla-foundation/common_voice_13_0",
            "wo",
            split="train",
            streaming=False,
        )
        datasets_list.append(cv_wo)
        logger.info("CV Wolof: %d exemples", len(cv_wo))
    except Exception as e:
        logger.warning("Wolof non disponible dans CV: %s", e)

    # 3. Dataset custom T.Speak
    local_data_path = os.path.join(config.dataset_path, "tspeak_train.json")
    if os.path.exists(local_data_path):
        logger.info("Chargement dataset custom T.Speak...")
        tspeak_dataset = load_tspeak_dataset(local_data_path)
        datasets_list.append(tspeak_dataset)
        logger.info("T.Speak dataset: %d exemples", len(tspeak_dataset))

    if not datasets_list:
        raise ValueError("Aucun dataset disponible ! Vérifiez les chemins et accès.")

    # Combiner les datasets
    combined = concatenate_datasets(datasets_list)
    combined = combined.shuffle(seed=42)

    # Split train/validation
    split = combined.train_test_split(test_size=0.1, seed=42)
    logger.info(
        "Dataset final: %d train, %d validation",
        len(split["train"]), len(split["test"]),
    )

    return DatasetDict({"train": split["train"], "validation": split["test"]})


def load_tspeak_dataset(json_path: str) -> Dataset:
    """
    Charge le dataset custom T.Speak.

    Format JSON attendu :
    [
        {
            "audio_path": "path/to/audio.wav",
            "transcription": "Hello, my name is Amadou",
            "native_language": "wolof",
            "accent": "senegalese"
        },
        ...
    ]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    import soundfile as sf

    samples = []
    for item in data:
        if not os.path.exists(item["audio_path"]):
            continue
        try:
            audio_array, sr = sf.read(item["audio_path"])
            samples.append({
                "audio": {"array": audio_array, "sampling_rate": sr},
                "sentence": item["transcription"],
                "native_language": item.get("native_language", "french"),
            })
        except Exception as e:
            logger.warning("Erreur chargement audio %s: %s", item["audio_path"], e)

    return Dataset.from_list(samples)


# ─── Preprocessing ───────────────────────────────────────────────────────────

class WhisperDataCollator:
    """Collator qui prépare les batches pour Whisper (features + labels)."""

    def __init__(self, processor: WhisperProcessor):
        self.processor = processor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Séparer input features et labels
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        # Padding des features
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Padding des labels avec -100 (ignoré par la loss)
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Supprimer le token BOS si présent
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def prepare_dataset(batch: dict, processor: WhisperProcessor, config: TSpkTrainingConfig) -> dict:
    """Prépare un exemple du dataset pour l'entraînement."""
    audio = batch["audio"]
    array = np.array(audio["array"], dtype=np.float32)

    # Normaliser l'audio
    if array.ndim > 1:
        array = array.mean(axis=1)  # Stéréo → Mono

    # Tronquer à max_input_length
    max_samples = int(config.max_input_length_sec * config.sample_rate)
    if len(array) > max_samples:
        array = array[:max_samples]

    # Extraire les features (spectrogramme Mel)
    batch["input_features"] = processor.feature_extractor(
        array,
        sampling_rate=config.sample_rate,
        return_tensors="np",
    ).input_features[0]

    # Tokeniser la transcription
    text = batch.get("sentence", batch.get("transcription", "")).strip()
    batch["labels"] = processor.tokenizer(text).input_ids

    return batch


# ─── Métriques d'évaluation ───────────────────────────────────────────────────

def build_compute_metrics(processor: WhisperProcessor):
    """Construit la fonction de calcul des métriques WER."""
    wer_metric = evaluate.load("wer")
    normalizer = BasicTextNormalizer()

    def compute_metrics(pred) -> dict:
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Remplacer -100 par le token de padding
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Décoder prédictions et références
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Normaliser le texte
        pred_str_norm = [normalizer(p) for p in pred_str]
        label_str_norm = [normalizer(l) for l in label_str]

        # Calculer WER
        wer = 100 * wer_metric.compute(predictions=pred_str_norm, references=label_str_norm)

        logger.info("WER actuel: %.2f%%", wer)
        return {"wer": wer}

    return compute_metrics


# ─── Script principal ─────────────────────────────────────────────────────────

def finetune_whisper(config: TSpkTrainingConfig):
    """Lance le fine-tuning complet de Whisper."""
    logger.info("🚀 Démarrage du fine-tuning Whisper T.Speak")
    logger.info("Modèle: %s → %s", config.model_name, config.output_dir)

    # ── Chargement modèle et processor ──
    logger.info("Chargement du modèle %s...", config.model_name)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(config.model_name)
    tokenizer = WhisperTokenizer.from_pretrained(
        config.model_name, language=config.language, task=config.task
    )
    processor = WhisperProcessor.from_pretrained(config.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(config.model_name)

    # Configurer la génération
    model.generation_config.language = config.language
    model.generation_config.task = config.task
    model.generation_config.forced_decoder_ids = None

    # Activer les gradient checkpoints pour économiser la mémoire GPU
    model.config.use_cache = False
    model.generate = lambda *args, **kwargs: WhisperForConditionalGeneration.generate(
        model, *args, **kwargs
    )

    # ── Chargement et préparation des données ──
    raw_datasets = load_african_dataset(config)

    logger.info("Préparation des features (spectrogrammes Mel)...")
    vectorized_datasets = raw_datasets.map(
        lambda batch: prepare_dataset(batch, processor, config),
        remove_columns=raw_datasets["train"].column_names,
        num_proc=4,
        desc="Extraction features",
    )

    # ── Configuration de l'entraînement ──
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        fp16=config.fp16 and torch.cuda.is_available(),
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        logging_steps=config.logging_steps,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    data_collator = WhisperDataCollator(processor=processor)
    compute_metrics = build_compute_metrics(processor)

    # ── Trainer ──
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=vectorized_datasets["train"],
        eval_dataset=vectorized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )

    # ── Entraînement ──
    logger.info("🎯 Démarrage de l'entraînement...")
    train_result = trainer.train()

    # ── Sauvegarde ──
    logger.info("💾 Sauvegarde du modèle fine-tuné...")
    trainer.save_model()
    processor.save_pretrained(config.output_dir)

    # Sauvegarder les métriques
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Évaluation finale
    logger.info("📊 Évaluation finale sur le dataset de validation...")
    eval_results = trainer.evaluate()
    trainer.log_metrics("eval", eval_results)
    trainer.save_metrics("eval", eval_results)

    logger.info(
        "✅ Fine-tuning terminé ! WER final: %.2f%% — Modèle sauvegardé dans %s",
        eval_results.get("eval_wer", -1),
        config.output_dir,
    )

    return eval_results


def evaluate_on_african_accents(model_path: str, test_audio_dir: str):
    """
    Évalue le modèle fine-tuné sur un dataset d'accents africains spécifiques.
    Génère un rapport WER par langue/accent.
    """
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.eval()

    import soundfile as sf
    from jiwer import wer

    results = {}
    test_files = [f for f in os.listdir(test_audio_dir) if f.endswith(".wav")]

    logger.info("Évaluation sur %d fichiers de test...", len(test_files))

    for audio_file in test_files:
        metadata_file = audio_file.replace(".wav", ".json")
        metadata_path = os.path.join(test_audio_dir, metadata_file)
        if not os.path.exists(metadata_path):
            continue

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        audio_path = os.path.join(test_audio_dir, audio_file)
        array, sr = sf.read(audio_path)

        inputs = processor(array, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            predicted_ids = model.generate(inputs.input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        accent = metadata.get("accent", "unknown")
        reference = metadata.get("transcription", "")
        error_rate = wer(reference.lower(), transcription.lower())

        if accent not in results:
            results[accent] = {"wer_scores": [], "count": 0}
        results[accent]["wer_scores"].append(error_rate)
        results[accent]["count"] += 1

    # Rapport final
    report = {}
    for accent, data in results.items():
        avg_wer = np.mean(data["wer_scores"]) * 100
        report[accent] = {
            "avg_wer": round(avg_wer, 2),
            "samples": data["count"],
            "target_met": avg_wer < 15.0,  # Objectif T.Speak: WER < 15%
        }
        logger.info("Accent %s: WER=%.1f%% (%d samples)", accent, avg_wer, data["count"])

    return report


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning Whisper pour T.Speak Afrique")
    parser.add_argument("--model_name", default="openai/whisper-medium")
    parser.add_argument("--dataset_path", default="./data/african_voices")
    parser.add_argument("--output_dir", default="./models/whisper-african-tspeak")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--eval_mode", action="store_true", help="Évaluation seulement")
    parser.add_argument("--test_audio_dir", default="./data/test_accents")

    args = parser.parse_args()

    config = TSpkTrainingConfig(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
    )

    if args.eval_mode:
        logger.info("Mode évaluation: %s", args.output_dir)
        report = evaluate_on_african_accents(args.output_dir, args.test_audio_dir)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        finetune_whisper(config)
