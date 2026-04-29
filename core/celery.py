"""T.Speak — Configuration Celery pour traitement audio asynchrone"""

import os
from celery import Celery
from celery.schedules import crontab

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")

app = Celery("tspeak")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()

# ─── Tâches périodiques ───────────────────────────────────────────────────────
app.conf.beat_schedule = {
    # Nettoyage des fichiers audio traités (RGPD)
    "cleanup-audio-files": {
        "task": "apps.sessions.tasks.cleanup_audio_files",
        "schedule": crontab(hour=2, minute=0),  # 2h du matin
    },
    # Mise à jour des classements
    "update-leaderboard": {
        "task": "apps.progress.tasks.update_leaderboard_cache",
        "schedule": crontab(minute="*/15"),  # Toutes les 15 minutes
    },
    # Reset des streaks à minuit (heure de Dakar)
    "reset-streaks": {
        "task": "apps.progress.tasks.check_streak_continuity",
        "schedule": crontab(hour=0, minute=5),
    },
    # Génération des rapports mensuels premium
    "generate-monthly-reports": {
        "task": "apps.scoring.tasks.generate_monthly_reports",
        "schedule": crontab(day_of_month=1, hour=6, minute=0),
    },
}

app.conf.timezone = "Africa/Dakar"


@app.task(bind=True, ignore_result=True)
def debug_task(self):
    print(f"Request: {self.request!r}")
