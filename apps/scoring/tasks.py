"""
T.Speak — Tâches Celery : Scoring & Rapports Mensuels
"""

import logging
from celery import shared_task
from django.core.cache import cache

logger = logging.getLogger(__name__)


@shared_task(name="apps.scoring.tasks.generate_monthly_reports")
def generate_monthly_reports():
    """
    Génère les rapports mensuels de progression pour chaque utilisateur premium.
    Exécuté le 1er de chaque mois à 06:00 (heure Dakar) par Celery Beat.
    """
    from datetime import timedelta
    from django.utils import timezone
    from django.contrib.auth import get_user_model
    from django.db.models import Avg, Count, Max

    User = get_user_model()
    now = timezone.now()
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    last_month_end = month_start - timedelta(seconds=1)
    last_month_start = last_month_end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    premium_users = User.objects.filter(is_active=True, is_premium=True)
    report_count = 0

    for user in premium_users:
        try:
            from apps.scoring.models import Score
            from apps.sessions.models import VocalSession

            monthly_scores = Score.objects.filter(
                user=user,
                created_at__gte=last_month_start,
                created_at__lte=last_month_end,
            )
            aggr = monthly_scores.aggregate(
                avg_pronunciation=Avg("pronunciation"),
                avg_fluency=Avg("fluency"),
                avg_grammar=Avg("grammar"),
                avg_vocabulary=Avg("vocabulary"),
                avg_global=Avg("global_score"),
                best_score=Max("global_score"),
                total_sessions=Count("id"),
            )
            sessions_count = VocalSession.objects.filter(
                user=user,
                status="completed",
                created_at__gte=last_month_start,
                created_at__lte=last_month_end,
            ).count()

            report = {
                "user_id": str(user.id),
                "period": f"{last_month_start.strftime('%Y-%m')}",
                "sessions": sessions_count,
                "scores": {k: round(float(v), 2) if v else 0 for k, v in aggr.items()},
                "generated_at": now.isoformat(),
            }

            cache_key = f"monthly_report:{user.id}:{last_month_start.strftime('%Y-%m')}"
            cache.set(cache_key, report, timeout=60 * 60 * 24 * 35)  # 35 jours
            report_count += 1

        except Exception as e:
            logger.error("Erreur rapport mensuel user=%s: %s", user.id, e, exc_info=True)

    logger.info("📊 Rapports mensuels générés: %d utilisateurs premium", report_count)
    return {"reports_generated": report_count, "period": last_month_start.strftime("%Y-%m")}
