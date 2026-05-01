"""
T.Speak — Tâches Celery : Gamification & Progression
Badges, streaks, classements — Pipeline de récompenses.
"""

import logging
from celery import shared_task
from django.core.cache import cache
from django.utils import timezone

logger = logging.getLogger(__name__)


# ─── Attribution des badges ───────────────────────────────────────────────────

@shared_task(
    name="apps.progress.tasks.check_and_award_badges",
    bind=True,
    max_retries=2,
    default_retry_delay=10,
)
def check_and_award_badges(self, user_id: str):
    """
    Vérifie et attribue les badges à un utilisateur après une session.
    Règles : première session, streaks 7j/30j, score ≥ 90, top 10.
    """
    try:
        from django.contrib.auth import get_user_model
        from apps.sessions.models import VocalSession
        from apps.users.models import Badge
        from django.conf import settings

        User = get_user_model()
        user = User.objects.get(id=user_id)
        badges_awarded = []

        # Get existing badge types
        current_badges = set(user.badges.values_list("badge_type", flat=True))
        badge_configs = getattr(settings, "BADGE_TYPES", {})

        def award(b_type):
            if b_type not in current_badges:
                conf = badge_configs.get(b_type, {"name": b_type, "icon": "🏅"})
                Badge.objects.get_or_create(
                    user=user,
                    badge_type=b_type,
                    defaults={
                        "badge_name": conf["name"],
                        "badge_icon": conf["icon"]
                    }
                )
                badges_awarded.append(b_type)
                logger.info("🏅 Badge '%s' décerné à %s", conf["name"], user.email)

        # Badge : Premier Pas
        if "first_session" not in current_badges:
            if VocalSession.objects.filter(user=user, status="completed").exists():
                award("first_session")

        # Badge : 7 jours de feu
        if "streak_7" not in current_badges and user.streak_days >= 7:
            award("streak_7")

        # Badge : Mois Parfait
        if "streak_30" not in current_badges and user.streak_days >= 30:
            award("streak_30")

        # Badge : Excellence (score ≥ 90)
        if "score_90" not in current_badges:
            from apps.scoring.models import Score
            if Score.objects.filter(user=user, global_score__gte=90).exists():
                award("score_90")

        # Badge : Elite (top 10 global leaderboard)
        if "top_10" not in current_badges:
            top_10_xp = list(
                User.objects.order_by("-xp_total").values_list("xp_total", flat=True)[:10]
            )
            if top_10_xp and user.xp_total >= top_10_xp[-1]:
                award("top_10")

        return {
            "status": "success",
            "user_id": user_id,
            "badges_awarded": badges_awarded,
        }

    except Exception as exc:
        logger.error("Erreur attribution badges user=%s: %s", user_id, exc, exc_info=True)
        try:
            raise self.retry(exc=exc)
        except Exception:
            return {"status": "error", "user_id": user_id, "badges_awarded": []}


# ─── Mise à jour du classement (périodique toutes les 15 min) ────────────────

@shared_task(name="apps.progress.tasks.update_leaderboard_cache")
def update_leaderboard_cache():
    """
    Recalcule et met en cache les classements hebdomadaire et global.
    Appelé toutes les 15 minutes par Celery Beat.
    """
    from datetime import timedelta
    from django.contrib.auth import get_user_model
    from apps.sessions.models import VocalSession

    User = get_user_model()

    # ── Classement Global : top 50 par XP total ────────────────────────────
    global_qs = (
        User.objects
        .filter(is_active=True)
        .order_by("-xp_total")
        .values("id", "first_name", "last_name", "xp_total", "streak_days", "sessions_count")[:50]
    )
    global_board = [
        {
            "rank": idx + 1,
            "user_id": str(u["id"]),
            "name": f"{u['first_name']} {u['last_name']}".strip(),
            "xp": u["xp_total"],
            "streak": u["streak_days"],
            "sessions": u["sessions_count"],
        }
        for idx, u in enumerate(global_qs)
    ]
    cache.set("leaderboard:global", global_board, timeout=60 * 20)  # 20 min

    # ── Classement Hebdomadaire : XP des 7 derniers jours ────────────────
    week_start = timezone.now() - timedelta(days=7)
    weekly_sessions = (
        VocalSession.objects
        .filter(status="completed", created_at__gte=week_start)
        .values("user_id", "user__first_name", "user__last_name")
    )

    weekly_xp: dict = {}
    for s in weekly_sessions:
        uid = str(s["user_id"])
        weekly_xp.setdefault(
            uid,
            {"name": f"{s['user__first_name']} {s['user__last_name']}".strip(), "xp": 0},
        )

    # Sum xp_earned per user for the week
    from django.db.models import Sum
    weekly_agg = (
        VocalSession.objects
        .filter(status="completed", created_at__gte=week_start)
        .values("user_id", "user__first_name", "user__last_name")
        .annotate(week_xp=Sum("xp_earned"))
        .order_by("-week_xp")[:50]
    )
    weekly_board = [
        {
            "rank": idx + 1,
            "user_id": str(row["user_id"]),
            "name": f"{row['user__first_name']} {row['user__last_name']}".strip(),
            "xp": row["week_xp"] or 0,
        }
        for idx, row in enumerate(weekly_agg)
    ]
    cache.set("leaderboard:weekly", weekly_board, timeout=60 * 20)  # 20 min

    logger.info(
        "🏆 Classements mis à jour: global=%d entrées, weekly=%d entrées",
        len(global_board),
        len(weekly_board),
    )
    return {"global": len(global_board), "weekly": len(weekly_board)}


# ─── Vérification des streaks à minuit ────────────────────────────────────────

@shared_task(name="apps.progress.tasks.check_streak_continuity")
def check_streak_continuity():
    """
    Réinitialise le streak des utilisateurs inactifs depuis plus de 24 heures.
    Exécuté à 00:05 heure de Dakar par Celery Beat.
    """
    from datetime import timedelta
    from django.contrib.auth import get_user_model

    User = get_user_model()
    cutoff = timezone.now() - timedelta(hours=24)

    # Utilisateurs avec un streak actif mais sans session récente
    inactive_users = User.objects.filter(streak_days__gt=0).exclude(
        sessions__created_at__gte=cutoff,
        sessions__status="completed",
    ).distinct()

    reset_count = inactive_users.count()
    inactive_users.update(streak_days=0)

    logger.info("🔄 Streaks réinitialisés pour %d utilisateurs inactifs", reset_count)
    return {"reset_count": reset_count}
