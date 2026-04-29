"""
T.Speak — Script Fine-Tuning Wav2Vec 2.0 pour Scoring Phonétique Africain
==========================================================================

Fine-tune wav2vec2-large-xlsr-53 pour reconnaître les phonèmes anglais
prononcés avec des accents africains francophones (Wolof, Pulaar, Bambara).

Objectif : améliorer la précision du scoring phonétique pour les accents locaux.

Usage:
    python scripts/finetune_wav2vec.py \
        --model_name facebook/wav2vec2-large-xlsr-53 \
        --dataset_path ./data/phoneme_data \
        --output_dir ./models/wav2vec-african-phonemes \
        --num_epochs 10
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import evaluate
import numpy as np
import torch
import torchaudio
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    EarlyStoppingCallback,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─── Vocabulaire phonémique étendu ───────────────────────────────────────────
# Inclut les phonèmes des langues locales pour améliorer le transfert d'apprentissage

VOCAB_DICT = {
    # Phonèmes anglais standard (ARPAbet)
    "AA": 0, "AE": 1, "AH": 2, "AO": 3, "AW": 4, "AY": 5,
    "B": 6, "CH": 7, "D": 8, "DH": 9, "EH": 10, "ER": 11, "EY": 12,
    "F": 13, "G": 14, "HH": 15, "IH": 16, "IY": 17, "JH": 18, "K": 19,
    "L": 20, "M": 21, "N": 22, "NG": 23, "OW": 24, "OY": 25, "P": 26,
    "R": 27, "S": 28, "SH": 29, "T": 30, "TH": 31, "UH": 32, "UW": 33,
    "V": 34, "W": 35, "Y": 36, "Z": 37, "ZH": 38,
    # Phonèmes wolof fréquents (pour transfert learning)
    "MB": 39, "ND": 40, "NG_INIT": 41,  # Consonnes prénasalisées wolof
    # Tokens spéciaux
    "|": 42,      # Séparateur de mots
    "[UNK]": 43,  # Inconnu
    "[PAD]": 44,  # Padding CTC
}


@dataclass
class Wav2VecTrainingConfig:
    """Configuration du fine-tuning Wav2Vec pour T.Speak."""
    model_name: str = "facebook/wav2vec2-large-xlsr-53"
    output_dir: str = "./models/wav2vec-african-phonemes"
    dataset_path: str = "./data/phoneme_data"

    # Hyperparamètres
    num_epochs: int = 10
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.005

    # CTC
    ctc_loss_reduction: str = "mean"
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    feat_proj_dropout: float = 0.0
    mask_time_prob: float = 0.05
    layerdrop: float = 0.0

    # Audio
    max_input_length_sec: float = 10.0
    sample_rate: int = 16000

    # Évaluation
    eval_steps: int = 400
    save_steps: int = 400
    logging_steps: int = 50


# ─── Préparation vocabulaire et tokenizer ────────────────────────────────────

def create_vocab_file(output_path: str):
    """Crée le fichier de vocabulaire phonémique."""
    vocab_reversed = {v: k for k, v in VOCAB_DICT.items()}
    vocab_json = {str(i): token for i, token in vocab_reversed.items()}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)
    logger.info("Vocabulaire créé: %d phonèmes", len(vocab_json))
    return output_path


def load_processor(config: Wav2VecTrainingConfig) -> Wav2Vec2Processor:
    """Crée le processor Wav2Vec avec notre vocabulaire phonémique."""
    vocab_path = os.path.join(config.output_dir, "vocab.json")
    create_vocab_file(vocab_path)

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=config.sample_rate,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )
    processor.save_pretrained(config.output_dir)
    return processor


# ─── Dataset de phonèmes ──────────────────────────────────────────────────────

def load_phoneme_dataset(config: Wav2VecTrainingConfig) -> DatasetDict:
    """
    Charge le dataset de phonèmes pour l'entraînement.

    Format JSON attendu pour T.Speak:
    [
        {
            "audio_path": "path/to/audio.wav",
            "phoneme_sequence": "HH EH L OW | W ER L D",
            "word_text": "Hello World",
            "native_language": "wolof",
            "difficulty": "TH,R,W"  // phonèmes difficiles dans cet enregistrement
        },
        ...
    ]
    """
    data_file = os.path.join(config.dataset_path, "phonemes.json")
    if not os.path.exists(data_file):
        logger.warning("Dataset %s introuvable — génération de données synthétiques", data_file)
        return _generate_synthetic_dataset()

    import soundfile as sf
    with open(data_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    samples = []
    for item in raw_data:
        if not os.path.exists(item["audio_path"]):
            continue
        try:
            array, sr = sf.read(item["audio_path"])
            if array.ndim > 1:
                array = array.mean(axis=1)
            # Resampler si nécessaire
            if sr != config.sample_rate:
                import librosa
                array = librosa.resample(array, orig_sr=sr, target_sr=config.sample_rate)

            samples.append({
                "audio": {"array": array.astype(np.float32), "sampling_rate": config.sample_rate},
                "phoneme_sequence": item["phoneme_sequence"],
                "word_text": item.get("word_text", ""),
                "native_language": item.get("native_language", "french"),
            })
        except Exception as e:
            logger.warning("Erreur chargement %s: %s", item["audio_path"], e)

    if not samples:
        raise ValueError("Aucun exemple valide dans le dataset !")

    dataset = Dataset.from_list(samples)
    dataset = dataset.shuffle(seed=42)
    split = dataset.train_test_split(test_size=0.15, seed=42)
    logger.info("Dataset phonèmes: %d train, %d test", len(split["train"]), len(split["test"]))
    return DatasetDict({"train": split["train"], "test": split["test"]})


def _generate_synthetic_dataset() -> DatasetDict:
    """
    Génère un mini-dataset synthétique pour tests.
    En production, remplacer par des vrais enregistrements.
    """
    logger.warning("⚠️  Utilisation de données synthétiques — qualité de production insuffisante")
    samples = []
    # Générer des exemples avec librosa pour test
    try:
        import librosa
        sr = 16000
        # Tons purs simulant des phonèmes
        for i, (phoneme_seq, duration) in enumerate([
            ("HH EH L OW", 1.0),
            ("TH AE NG K | Y UW", 1.2),
            ("M AY | N EY M | IH Z", 1.5),
            ("DH AH | W EH DH ER | IH Z | N AY S", 2.0),
        ]):
            t = np.linspace(0, duration, int(sr * duration))
            freq = 440 + i * 100
            audio = (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)
            samples.append({
                "audio": {"array": audio, "sampling_rate": sr},
                "phoneme_sequence": phoneme_seq,
                "word_text": f"example_{i}",
                "native_language": "french",
            })
    except ImportError:
        pass

    if not samples:
        raise ValueError("Impossible de générer des données synthétiques.")

    dataset = Dataset.from_list(samples)
    split = dataset.train_test_split(test_size=0.25, seed=42)
    return DatasetDict({"train": split["train"], "test": split["test"]})


# ─── Preprocessing ───────────────────────────────────────────────────────────

def prepare_dataset_wav2vec(
    batch: dict,
    processor: Wav2Vec2Processor,
    config: Wav2VecTrainingConfig,
) -> dict:
    """Prépare les features et labels pour Wav2Vec2 CTC."""
    audio = batch["audio"]
    array = np.array(audio["array"], dtype=np.float32)

    # Normalisation audio
    if np.abs(array).max() > 0:
        array = array / np.abs(array).max()

    # Tronquer si trop long
    max_samples = int(config.max_input_length_sec * config.sample_rate)
    array = array[:max_samples]

    # Extraire les features
    batch["input_values"] = processor(
        array, sampling_rate=config.sample_rate
    ).input_values[0]

    # Encoder les phonèmes comme labels
    with processor.as_target_processor():
        batch["labels"] = processor(batch["phoneme_sequence"]).input_ids

    return batch


@dataclass
class DataCollatorCTCWithPadding:
    """Collator avec padding pour CTC (longueurs variables)."""

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


# ─── Métriques ───────────────────────────────────────────────────────────────

def build_phoneme_metrics(processor: Wav2Vec2Processor):
    """Métriques : PER (Phone Error Rate) et WER."""
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred) -> dict:
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)

        per = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        logger.info("PER: %.2f%%", per)
        return {"per": per}

    return compute_metrics


# ─── Fine-tuning principal ────────────────────────────────────────────────────

def finetune_wav2vec(config: Wav2VecTrainingConfig):
    """Lance le fine-tuning Wav2Vec2 pour le scoring phonétique T.Speak."""
    logger.info("🎵 Démarrage fine-tuning Wav2Vec2 pour T.Speak")

    os.makedirs(config.output_dir, exist_ok=True)

    # ── Processor ──
    processor = load_processor(config)

    # ── Dataset ──
    raw_datasets = load_phoneme_dataset(config)
    logger.info("Préparation des features audio...")
    processed_datasets = raw_datasets.map(
        lambda b: prepare_dataset_wav2vec(b, processor, config),
        remove_columns=raw_datasets["train"].column_names,
        num_proc=2,
        desc="Feature extraction",
    )

    # ── Modèle ──
    logger.info("Chargement Wav2Vec2: %s", config.model_name)
    model = Wav2Vec2ForCTC.from_pretrained(
        config.model_name,
        attention_dropout=config.attention_dropout,
        hidden_dropout=config.hidden_dropout,
        feat_proj_dropout=config.feat_proj_dropout,
        mask_time_prob=config.mask_time_prob,
        layerdrop=config.layerdrop,
        ctc_loss_reduction=config.ctc_loss_reduction,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,  # La tête CTC a une taille différente
    )

    # Geler le feature extractor CNN pour les premières epochs
    model.freeze_feature_extractor()

    # ── Training args ──
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        group_by_length=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        fp16=torch.cuda.is_available(),
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        evaluation_strategy="steps",
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        logging_steps=config.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="per",
        greater_is_better=False,
        save_total_limit=3,
        push_to_hub=False,
        report_to=["tensorboard"],
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    compute_metrics = build_phoneme_metrics(processor)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_datasets["test"],
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ── Entraînement ──
    logger.info("🎯 Démarrage de l'entraînement phonémique...")
    trainer.train()

    # ── Sauvegarde ──
    logger.info("💾 Sauvegarde du modèle...")
    trainer.save_model()
    processor.save_pretrained(config.output_dir)

    # Évaluation finale
    eval_results = trainer.evaluate()
    logger.info("✅ Fine-tuning terminé ! PER final: %.2f%%", eval_results.get("eval_per", -1))

    # Sauvegarder la config T.Speak
    tspeak_config = {
        "model_type": "wav2vec2-phoneme-scorer",
        "base_model": config.model_name,
        "target_languages": ["en"],
        "supported_native_languages": ["wolof", "pulaar", "bambara", "french"],
        "phoneme_vocab_size": len(processor.tokenizer),
        "eval_per": eval_results.get("eval_per", -1),
        "training_config": {
            "num_epochs": config.num_epochs,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
        },
    }
    with open(os.path.join(config.output_dir, "tspeak_config.json"), "w") as f:
        json.dump(tspeak_config, f, indent=2)

    return eval_results


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning Wav2Vec2 pour T.Speak")
    parser.add_argument("--model_name", default="facebook/wav2vec2-large-xlsr-53")
    parser.add_argument("--dataset_path", default="./data/phoneme_data")
    parser.add_argument("--output_dir", default="./models/wav2vec-african-phonemes")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)

    args = parser.parse_args()

    config = Wav2VecTrainingConfig(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    finetune_wav2vec(config)
