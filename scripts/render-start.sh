#!/bin/bash

# Arrêt immédiat en cas d'erreur
set -e

echo "🚀 Démarrage de TSpeaker Backend sur Render..."

# 1. Attente facultative de la DB (Render gère ça via les dépendances de Blueprint)

# 2. Application des migrations
echo "⚙️ Application des migrations Django..."
python manage.py migrate --noinput

# 3. Collecte des fichiers statiques
echo "📦 Collecte des fichiers statiques..."
python manage.py collectstatic --noinput

# 4. Lancement de Gunicorn
# On utilise $PORT fourni par Render ou 8000 par défaut
echo "🌍 Lancement de Gunicorn sur le port ${PORT:-8000}..."
exec gunicorn core.wsgi:application \
    --bind 0.0.0.0:${PORT:-8000} \
    --workers 3 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile -
