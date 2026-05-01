import os
import torch
from datasets import load_dataset
from transformers import (
    WhisperForConditionalGeneration, 
    WhisperProcessor, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    WhisperTokenizer,
    WhisperFeatureExtractor
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def finetune_lora(model_id="openai/whisper-medium", output_dir="./whisper-lora"):
    """
    Exemple de fine-tuning LoRA pour Whisper.
    Plus efficace, moins de mémoire, pas d'oubli catastrophique.
    """
    # 1. Charger modèle et processor
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    
    # 2. Configurer LoRA
    config = LoraConfig(
        r=32, 
        lora_alpha=64, 
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05, 
        bias="none"
    )
    
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    
    # Note: On ne lance pas l'entraînement ici car il faut le dataset préparé
    print(f"Modèle LoRA prêt. Prêt à être entraîné sur {output_dir}")
    
    # Sauvegarde du squelette de config
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)

if __name__ == "__main__":
    finetune_lora()
