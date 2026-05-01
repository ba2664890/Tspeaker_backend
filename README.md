# T.Speak — Backend Django + Moteur IA Vocal
> Apprendre l'anglais par la voix — Fait pour l'Afrique 🌍

Backend complet pour l'application mobile T.Speak :  
**Django REST API + Celery + Whisper + Wav2Vec 2.0 + LLM**

---

## Architecture

```
tspeak_backend/
├── core/                    # Configuration Django centrale
│   ├── settings.py          # Settings complets (DB, Redis, IA, sécurité)
│   ├── urls.py              # Routes principales
│   ├── celery.py            # Celery + tâches périodiques
│   └── middleware.py        # Logging, exceptions custom
│
├── apps/
│   ├── users/               # Auth JWT, profil, leaderboard
│   │   ├── models.py        # User étendu + Badge + streaks
│   │   ├── serializers.py   # RegisterSerializer, ProfileSerializer
│   │   └── views.py         # Register, Login, Logout, Profil, Streak
│   ├── sessions/            # Sessions vocales
│   │   ├── models.py        # VocalSession + AudioExchange
│   │   ├── views.py         # Start, Upload audio, History, End
│   │   └── tasks.py         # Pipeline Celery : Whisper→Wav2Vec→LLM
│   ├── scoring/             # Scores et statistiques
│   │   └── models.py        # Score (prononciation/fluidité/grammaire/vocab)
│   ├── simulations/         # Simulations pro (Pitch, Entretien...)
│   │   └── models.py        # Simulation + vues ListAPI/Detail/Start
│   └── progress/            # Gamification et progression
│
├── ai/
│   ├── whisper_asr/
│   │   └── transcriber.py   # WhisperTranscriber : transcription + WER
│   ├── wav2vec_scoring/
│   │   ├── scorer.py        # Wav2VecScorer : phonèmes + scoring
│   │   └── nlp_analyzer.py  # GrammarAnalyzer + VocabularyAnalyzer
│   └── llm_conversation/
│       └── generator.py     # ConversationGenerator : feedback + questions
│
├── scripts/
│   ├── finetune_whisper.py  # Fine-tuning Whisper sur accents africains
│   └── finetune_wav2vec.py  # Fine-tuning Wav2Vec sur phonèmes africains
│
├── tests/
│   └── test_backend.py      # Tests unitaires + intégration (>80% couverture)
│
├── docker-compose.yml       # Stack complète (API, Worker, DB, Redis, Nginx)
└── requirements.txt         # Dépendances Python
```

---

## Installation rapide

### 1. Pré-requis
```bash
python 3.11+
postgresql 15+
redis 7+
ffmpeg  # Pour conversion audio
```

### 2. Environnement
```bash
git clone <repo>
cd tspeak_backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configurer .env
cp .env.example .env
# Éditer : SECRET_KEY, DB_PASSWORD, LLM_API_KEY
```

### 3. Base de données
```bash
python manage.py migrate
python manage.py createsuperuser
python manage.py loaddata fixtures/simulations.json  # Données de démo
```

### 4. Lancer les services
```bash
# Terminal 1 — Django API
python manage.py runserver

# Terminal 2 — Celery Worker (traitement audio)
celery -A core.celery worker --loglevel=info --queues=audio,default

# Terminal 3 — Celery Beat (tâches périodiques)
celery -A core.celery beat --loglevel=info
```

### 5. Docker (recommandé en production)
```bash
docker-compose up -d
docker-compose logs -f tspeak-api
```

---

## Pipeline Audio (< 3 secondes)

```
Flutter  →  POST /api/v1/sessions/{id}/audio/  (WAV/M4A, max 10MB)
              ↓
         Django valide le fichier
              ↓
         Celery reçoit la tâche (Redis broker)
              ↓
         FFmpeg → WAV 16kHz mono
              ↓
         Whisper transcrit (< 1.5s pour 30s d'audio)
              ↓
         Wav2Vec analyse phonèmes par phonème
              ↓
         NLP : grammaire + vocabulaire (spaCy)
              ↓
         LLM génère feedback + prochaine question
              ↓
         Score calculé → PostgreSQL
              ↓
         Résultat mis en cache Redis (TTL 10min)
              ↓
Flutter  ←  GET /api/v1/sessions/exchanges/{id}/result/
```

---

## Endpoints API principaux

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| POST | `/api/v1/auth/register/` | Inscription |
| POST | `/api/v1/auth/login/` | Connexion JWT |
| POST | `/api/v1/auth/logout/` | Déconnexion |
| GET/PATCH | `/api/v1/auth/me/` | Profil utilisateur |
| POST | `/api/v1/sessions/start/` | Démarrer session |
| POST | `/api/v1/sessions/{id}/audio/` | Upload audio |
| GET | `/api/v1/sessions/exchanges/{id}/result/` | Résultat IA |
| GET | `/api/v1/sessions/history/` | Historique |
| GET | `/api/v1/simulations/` | Liste simulations |
| POST | `/api/v1/simulations/{id}/start/` | Démarrer simulation |
| GET | `/api/v1/auth/leaderboard/` | Classement |

Documentation interactive : `http://localhost:8000/api/docs/`

---

## Fine-Tuning des Modèles IA

### Whisper (accents africains)
```bash
python scripts/finetune_whisper.py \
    --model_name openai/whisper-medium \
    --dataset_path ./data/african_voices \
    --output_dir ./models/whisper-african \
    --num_epochs 5 \
    --batch_size 16

# Évaluation WER par accent
python scripts/finetune_whisper.py --eval_mode \
    --output_dir ./models/whisper-african \
    --test_audio_dir ./data/test_accents
```

### Wav2Vec 2.0 (phonèmes africains)
```bash
python scripts/finetune_wav2vec.py \
    --model_name facebook/wav2vec2-large-xlsr-53 \
    --dataset_path ./data/phoneme_data \
    --output_dir ./models/wav2vec-african-phonemes \
    --num_epochs 10
```

### Format des données d'entraînement
```json
// data/african_voices/tspeak_train.json
[
  {
    "audio_path": "audio/dakar_001.wav",
    "transcription": "I would like to present my startup",
    "native_language": "wolof",
    "accent": "senegalese"
  }
]
```

---

## Tests

```bash
# Tests complets
pytest tests/ -v --cov=apps --cov-report=html --cov-report=term

# Tests par module
pytest tests/test_backend.py::TestAuthentication -v
pytest tests/test_backend.py::TestWhisperTranscriber -v
pytest tests/test_backend.py::TestGrammarAnalyzer -v

# Tests de performance (Locust)
locust -f tests/locustfile.py --host=http://localhost:8000
```

Couverture cible : **> 80%** ✅

---

## Variables d'environnement

```bash
# Django
SECRET_KEY=your-secret-key-here
DEBUG=False
ALLOWED_HOSTS=localhost,api.tspeak.africa

# Base de données
DB_NAME=tspeak_db
DB_USER=tspeak_user
DB_PASSWORD=secure_password
DB_HOST=localhost

# Redis
REDIS_URL=redis://localhost:6379

# IA Models
WHISPER_MODEL=medium          # tiny/base/small/medium/large
WHISPER_DEVICE=cpu            # cpu/cuda
WHISPER_COMPUTE_TYPE=default  # default/int8/float16
WHISPER_FINE_TUNED_PATH=./models/whisper-african  # Optionnel
WAV2VEC_MODEL=facebook/wav2vec2-base-960h
WAV2VEC_DEVICE=cpu            # cpu/cuda
WAV2VEC_MAX_AUDIO_SECONDS=120
AUDIO_PROCESSING_MODE=async   # async/sync
AUDIO_INLINE_FALLBACK_ENABLED=True
AUDIO_INLINE_FALLBACK_AFTER_SECONDS=8
LLM_API_KEY=sk-...            # OpenAI ou compatible

# Stockage (optionnel, sinon local)
USE_S3=False
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_STORAGE_BUCKET_NAME=tspeak-audio

# Monitoring
SENTRY_DSN=https://...
```

---

## Métriques de qualité cibles

| Métrique | Objectif | Description |
|----------|----------|-------------|
| WER Whisper (accents africains) | < 15% | Word Error Rate |
| Latence transcription | < 1.5s | Pour 30s d'audio |
| Latence feedback complet | < 3s | P95 |
| Couverture tests backend | > 80% | pytest-cov |
| Disponibilité API | > 99.5% | Uptime mensuel |

---

*T.Speak — Fait en Afrique, pour l'Afrique 🌍*  
*Projet de Fin d'Études — Deep Learning — Avril 2026*
# Tspeaker_backend
