# ─── Build Stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Installation des dépendances système nécessaires à la compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    python3-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ─── Final Stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

WORKDIR /app

# Installation des dépendances d'exécution (FFmpeg est CRITIQUE pour l'IA audio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Récupération des packages installés dans le builder
COPY --from=builder /install /usr/local

# Copie du code source
COPY . .

# Création des répertoires nécessaires et gestion des permissions
RUN mkdir -p /app/staticfiles /app/media /app/logs /tmp/tspeak_audio \
    && chmod +x /app/scripts/render-start.sh

# L'exposition du port est indicative (Render utilise la variable $PORT)
EXPOSE 8000

# Par défaut, on lance le serveur web, mais cela sera écrasé par render.yaml pour les workers
CMD ["/app/scripts/render-start.sh"]
