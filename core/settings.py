"""
T.Speak — Django Settings
Application d'apprentissage de l'anglais par IA vocale pour l'Afrique
"""

import os
from pathlib import Path
from datetime import timedelta
from decouple import config, Csv

BASE_DIR = Path(__file__).resolve().parent.parent

# ─── Sécurité ───────────────────────────────────────────────────────────────
SECRET_KEY = config("SECRET_KEY", default="change-me-in-production-please-longer-key-32-chars")
DEBUG = config("DEBUG", default=False, cast=bool)
ALLOWED_HOSTS = config(
    "ALLOWED_HOSTS", 
    default="localhost,127.0.0.1,tspeaker-backend-1.onrender.com", 
    cast=Csv()
)
# Ajout des domaines Render dynamiquement
RENDER_EXTERNAL_HOSTNAME = os.environ.get("RENDER_EXTERNAL_HOSTNAME")
if RENDER_EXTERNAL_HOSTNAME:
    ALLOWED_HOSTS.append(RENDER_EXTERNAL_HOSTNAME)

CSRF_TRUSTED_ORIGINS = [
    "https://tspeaker-backend-1.onrender.com",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

# ─── Applications ────────────────────────────────────────────────────────────
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # Third-party
    "rest_framework",
    "rest_framework_simplejwt",
    "rest_framework_simplejwt.token_blacklist",
    "corsheaders",
    "django_extensions",
    "drf_spectacular",
    # T.Speak apps
    "apps.users",
    "apps.sessions",
    "apps.scoring",
    "apps.simulations",
    "apps.content",
    "apps.progress",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "core.middleware.RequestLoggingMiddleware",
]

ROOT_URLCONF = "core.urls"
WSGI_APPLICATION = "core.wsgi.application"
AUTH_USER_MODEL = "users.User"

# ─── Templates ───────────────────────────────────────────────────────────────
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

import dj_database_url

# ─── Base de données ─────────────────────────────────────────────────────────
DATABASES = {
    "default": dj_database_url.config(
        default=f'sqlite:///{BASE_DIR / "db.sqlite3"}',
        conn_max_age=600,
        conn_health_checks=True,
    )
}

# ─── Cache & Redis ───────────────────────────────────────────────────────────
REDIS_URL = config("REDIS_URL", default="redis://127.0.0.1:6379")
CACHE_BACKEND = config("CACHE_BACKEND", default="redis")

# Nettoyage du REDIS_URL pour éviter les doubles slashes lors de l'ajout de l'index
_cleaned_redis_url = REDIS_URL.rstrip('/')

if CACHE_BACKEND == "locmem":
    CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
            "LOCATION": "tspeak-default-cache",
            "TIMEOUT": 300,
        },
        "sessions": {
            "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
            "LOCATION": "tspeak-session-cache",
        },
    }
else:
    CACHES = {
        "default": {
            "BACKEND": "django_redis.cache.RedisCache",
            "LOCATION": f"{_cleaned_redis_url}/0",
            "OPTIONS": {
                "CLIENT_CLASS": "django_redis.client.DefaultClient",
                "CONNECTION_POOL_KWARGS": {"max_connections": 20},
                "SOCKET_CONNECT_TIMEOUT": 5,
                "SOCKET_TIMEOUT": 5,
            },
            "KEY_PREFIX": "tspeak",
            "TIMEOUT": 300,
        },
        "sessions": {
            "BACKEND": "django_redis.cache.RedisCache",
            "LOCATION": f"{_cleaned_redis_url}/1",
            "OPTIONS": {
                "CLIENT_CLASS": "django_redis.client.DefaultClient",
            },
            "KEY_PREFIX": "tspeak_sess",
            "TIMEOUT": 86400, # 24h
        },
    }

SESSION_ENGINE = "django.contrib.sessions.backends.cache"
SESSION_CACHE_ALIAS = "sessions"

# ─── Celery ──────────────────────────────────────────────────────────────────
CELERY_BROKER_URL = f"{_cleaned_redis_url}/2"
CELERY_RESULT_BACKEND = f"{_cleaned_redis_url}/2"
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_TIMEZONE = "Africa/Dakar"
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_TIME_LIMIT = 300  # 5 min max par tâche audio
CELERY_WORKER_CONCURRENCY = 4

# ─── DRF ────────────────────────────────────────────────────────────────────
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": (
        "rest_framework_simplejwt.authentication.JWTAuthentication",
    ),
    "DEFAULT_PERMISSION_CLASSES": ("rest_framework.permissions.IsAuthenticated",),
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 20,
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.AnonRateThrottle",
        "rest_framework.throttling.UserRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "anon": "30/min",
        "user": "100/min",
        "login": "12/min",
        "register": "10/hour",
        "audio_upload": "10/min",
    },
    "EXCEPTION_HANDLER": "core.exceptions.custom_exception_handler",
}

# ─── JWT ────────────────────────────────────────────────────────────────────
SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(minutes=15),
    "REFRESH_TOKEN_LIFETIME": timedelta(days=7),
    "ROTATE_REFRESH_TOKENS": True,
    "BLACKLIST_AFTER_ROTATION": True,
    "ALGORITHM": "HS256",
    "SIGNING_KEY": SECRET_KEY,
    "AUTH_HEADER_TYPES": ("Bearer",),
    "TOKEN_BLACKLIST_SERIALIZER": "rest_framework_simplejwt.serializers.TokenBlacklistSerializer",
}

# ─── CORS ────────────────────────────────────────────────────────────────────
CORS_ALLOWED_ORIGINS = config(
    "CORS_ALLOWED_ORIGINS",
    default="http://localhost:3000,http://localhost:8080,http://127.0.0.1:3000,http://127.0.0.1:8080",
    cast=Csv(),
)
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_HEADERS = [
    "accept", "accept-encoding", "authorization",
    "content-type", "dnt", "origin", "user-agent",
    "x-csrftoken", "x-requested-with",
]

# ─── Stockage fichiers ───────────────────────────────────────────────────────
USE_S3 = config("USE_S3", default=False, cast=bool)

if USE_S3:
    DEFAULT_FILE_STORAGE = "storages.backends.s3boto3.S3Boto3Storage"
    AWS_ACCESS_KEY_ID = config("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = config("AWS_SECRET_ACCESS_KEY")
    AWS_STORAGE_BUCKET_NAME = config("AWS_STORAGE_BUCKET_NAME", default="tspeak-audio")
    AWS_S3_REGION_NAME = config("AWS_S3_REGION_NAME", default="af-south-1")
    AWS_S3_FILE_OVERWRITE = False
    AWS_DEFAULT_ACL = "private"
    AWS_S3_OBJECT_PARAMETERS = {"ContentDisposition": "attachment"}
else:
    MEDIA_URL = "/media/"
    MEDIA_ROOT = BASE_DIR / "media"

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

# ─── Configuration Audio ─────────────────────────────────────────────────────
AUDIO_MAX_SIZE_MB = 10
AUDIO_MAX_DURATION_SEC = 120
AUDIO_ALLOWED_FORMATS = ["wav", "m4a", "mp3", "ogg", "webm"]
AUDIO_TARGET_SAMPLE_RATE = 16000  # Whisper exige 16kHz
AUDIO_DELETE_AFTER_PROCESSING = True  # RGPD : suppression après 24h

# ─── Configuration IA ────────────────────────────────────────────────────────
WHISPER_MODEL = config("WHISPER_MODEL", default="medium")
WHISPER_DEVICE = config("WHISPER_DEVICE", default="cpu")  # "cuda" pour GPU
WHISPER_FINE_TUNED_PATH = config("WHISPER_FINE_TUNED_PATH", default=None)
WAV2VEC_MODEL = config("WAV2VEC_MODEL", default="facebook/wav2vec2-large-xlsr-53")
LLM_API_KEY = config("LLM_API_KEY", default="")
LLM_MODEL = config("LLM_MODEL", default="gpt-4o-mini")
LLM_MAX_TOKENS = 500
AI_PROCESSING_TIMEOUT = 30  # secondes

# ─── Gamification ────────────────────────────────────────────────────────────
XP_PER_SESSION = 50
XP_BONUS_PERFECT = 100
XP_PER_SIMULATION = 150
STREAK_RESET_HOUR = 0  # Minuit heure locale

BADGE_TYPES = {
    "first_session": {"name": "Premier Pas", "icon": "🎯"},
    "streak_7": {"name": "7 Jours de Feu", "icon": "🔥"},
    "streak_30": {"name": "Mois Parfait", "icon": "💎"},
    "score_90": {"name": "Excellence", "icon": "⭐"},
    "simulation_pitch": {"name": "Pitch Master", "icon": "💼"},
    "wolof_bridge": {"name": "Pont Culturel", "icon": "🌍"},
    "top_10": {"name": "Elite", "icon": "🏆"},
}

# ─── DRF Spectacular (OpenAPI) ────────────────────────────────────────────────
SPECTACULAR_SETTINGS = {
    "TITLE": "T.Speak API",
    "DESCRIPTION": "Backend IA pour l'apprentissage de l'anglais oral en Afrique francophone",
    "VERSION": "1.0.0",
    "SERVE_INCLUDE_SCHEMA": False,
    "COMPONENT_SPLIT_REQUEST": True,
    "CONTACT": {"name": "T.Speak Team", "email": "dev@tspeak.africa"},
}

# ─── Logging ─────────────────────────────────────────────────────────────────
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "[{asctime}] {levelname} {name} {module}: {message}",
            "style": "{",
        },
        "simple": {"format": "{levelname} {message}", "style": "{"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": BASE_DIR / "logs" / "tspeak.log",
            "maxBytes": 10 * 1024 * 1024,
            "backupCount": 5,
            "formatter": "verbose",
        },
    },
    "root": {"handlers": ["console", "file"], "level": "INFO"},
    "loggers": {
        "django": {"handlers": ["console"], "level": "WARNING", "propagate": False},
        "tspeak.ai": {"handlers": ["console", "file"], "level": "DEBUG", "propagate": False},
        "tspeak.auth": {"handlers": ["console", "file"], "level": "INFO", "propagate": False},
    },
}

# ─── Internationalisation ────────────────────────────────────────────────────
LANGUAGE_CODE = "fr-sn"
TIME_ZONE = "Africa/Dakar"
USE_I18N = True
USE_TZ = True
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ─── Sentry ──────────────────────────────────────────────────────────────────
SENTRY_DSN = config("SENTRY_DSN", default="")
if SENTRY_DSN:
    import sentry_sdk
    from sentry_sdk.integrations.django import DjangoIntegration
    from sentry_sdk.integrations.celery import CeleryIntegration

    sentry_sdk.init(
        dsn=SENTRY_DSN,
        integrations=[DjangoIntegration(), CeleryIntegration()],
        traces_sample_rate=0.1,
        send_default_pii=False,
    )
