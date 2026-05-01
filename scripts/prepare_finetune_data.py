import os
import django
import json
import httpx
import logging
from pathlib import Path
from tqdm import tqdm

# Configuration de Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
django.setup()

from apps.sessions.models import AudioExchange

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/finetune")
AUDIO_DIR = DATA_DIR / "audio"
DATA_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

def export_from_db(output_json="tspeak_train.json"):
    """Exporte les données de la table AudioExchange vers un format JSON pour le fine-tuning."""
    exchanges = AudioExchange.objects.exclude(transcription="").exclude(user_audio_url="")
    logger.info(f"Trouvé {exchanges.count()} échanges avec transcription.")
    
    data = []
    with httpx.Client() as client:
        for ex in tqdm(exchanges, desc="Exportation"):
            # Téléchargement de l'audio si nécessaire
            audio_filename = f"{ex.id}.wav"
            audio_path = AUDIO_DIR / audio_filename
            
            if not audio_path.exists():
                try:
                    resp = client.get(ex.user_audio_url)
                    resp.raise_for_status()
                    audio_path.write_bytes(resp.content)
                except Exception as e:
                    logger.warning(f"Impossible de télécharger {ex.user_audio_url}: {e}")
                    continue
            
            # Récupération de l'indice de langue via la session
            native_lang = ex.session.native_language_hint or "unknown"
            
            data.append({
                "audio_path": str(audio_path.absolute()),
                "transcription": ex.transcription,
                "language": "en", # Cible : anglais
                "native_language": native_lang,
                "exchange_id": str(ex.id)
            })
            
    output_path = DATA_DIR / output_json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Dataset exporté vers {output_path} ({len(data)} échantillons)")

def fetch_external_wolof():
    """Exemple de fetch de données Wolof externes via HuggingFace (si besoin)."""
    from datasets import load_dataset
    logger.info("Chargement de FLEURS Wolof...")
    try:
        dataset = load_dataset("google/fleurs", "wo_sn", split="train", streaming=True)
        # On pourrait en extraire quelques-uns pour le test
        logger.info("FLEURS Wolof accessible en streaming.")
    except Exception as e:
        logger.error(f"Erreur FLEURS: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["db", "external"], default="db")
    args = parser.parse_args()
    
    if args.mode == "db":
        export_from_db()
    else:
        fetch_external_wolof()
