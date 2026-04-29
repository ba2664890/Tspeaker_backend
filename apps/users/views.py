"""T.Speak — Vues Authentification & Profil utilisateur"""

import logging
from datetime import timedelta

from django.core.cache import cache
from django.db.models import Count, IntegerField, Q, Sum, Value
from django.db.models.functions import Coalesce
from django.utils import timezone
from rest_framework import generics, status, permissions
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.exceptions import TokenError
from drf_spectacular.utils import extend_schema, OpenApiParameter

from .models import User, Badge
from .serializers import (
    RegisterSerializer,
    TSpkTokenObtainSerializer,
    UserProfileSerializer,
    UserProfileUpdateSerializer,
    BadgeSerializer,
    LeaderboardResponseSerializer,
)

logger = logging.getLogger("tspeak.auth")


LEVEL_MAP = {
    "beginner": 1,
    "elementary": 2,
    "intermediate": 3,
    "upper_intermediate": 4,
    "advanced": 5,
}

LEAGUE_TIERS = [
    {"name": "Éclosion", "min_xp": 0},
    {"name": "Impulsion", "min_xp": 500},
    {"name": "Pionnier", "min_xp": 1500},
    {"name": "Virtuose", "min_xp": 3500},
    {"name": "Élite", "min_xp": 7500},
    {"name": "Légende", "min_xp": 12000},
]


def _user_display_name(row):
    name = f"{row.get('first_name', '')} {row.get('last_name', '')}".strip()
    if name:
        return name

    email = row.get("email", "")
    if email:
        return email.split("@", 1)[0]

    return "Apprenant T.Speak"


def _league_meta(total_xp):
    current = LEAGUE_TIERS[0]
    next_tier = None

    for index, tier in enumerate(LEAGUE_TIERS):
        if total_xp >= tier["min_xp"]:
            current = tier
            next_tier = LEAGUE_TIERS[index + 1] if index + 1 < len(LEAGUE_TIERS) else None
        else:
            break

    current_floor = current["min_xp"]
    next_target = next_tier["min_xp"] if next_tier else total_xp

    if not next_tier or next_target <= current_floor:
        progress = 1.0
        remaining = 0
    else:
        progress = (total_xp - current_floor) / (next_target - current_floor)
        progress = max(0.0, min(progress, 1.0))
        remaining = max(0, next_target - total_xp)

    return {
        "current_league": current["name"],
        "next_league": next_tier["name"] if next_tier else "",
        "league_progress": round(progress, 4),
        "next_league_target": next_target,
        "score_to_next_league": remaining,
    }


def _scope_meta(scope):
    if scope == "global":
        return {
            "scope": "global",
            "scope_label": "Global",
            "scope_description": "Le classement cumulé sur toute l'aventure T.Speak.",
            "score_label": "XP total",
        }

    return {
        "scope": "weekly",
        "scope_label": "Hebdomadaire",
        "scope_description": "Les XP gagnés sur les 7 derniers jours.",
        "score_label": "XP semaine",
    }


def _build_leaderboard_snapshot(scope):
    cache_key = f"leaderboard:{scope}"
    cached_snapshot = cache.get(cache_key)
    if cached_snapshot:
        return cached_snapshot

    queryset = User.objects.filter(is_active=True)

    if scope == "global":
        queryset = queryset.annotate(
            ranking_xp=Coalesce("xp_total", Value(0), output_field=IntegerField()),
            activity_sessions=Coalesce("sessions_count", Value(0), output_field=IntegerField()),
        )
    else:
        cutoff = timezone.now() - timedelta(days=7)
        queryset = queryset.annotate(
            ranking_xp=Coalesce(
                Sum(
                    "sessions__xp_earned",
                    filter=Q(
                        sessions__status="completed",
                        sessions__created_at__gte=cutoff,
                    ),
                ),
                Value(0),
                output_field=IntegerField(),
            ),
            activity_sessions=Coalesce(
                Count(
                    "sessions",
                    filter=Q(
                        sessions__status="completed",
                        sessions__created_at__gte=cutoff,
                    ),
                    distinct=True,
                ),
                Value(0),
                output_field=IntegerField(),
            ),
        )

    raw_rows = queryset.values(
        "id",
        "email",
        "first_name",
        "last_name",
        "avatar_url",
        "xp_total",
        "level",
        "streak_days",
        "sessions_count",
        "avg_pronunciation",
        "avg_fluency",
        "avg_grammar",
        "avg_vocabulary",
        "ranking_xp",
        "activity_sessions",
        "date_joined",
    ).order_by("-ranking_xp", "-streak_days", "-activity_sessions", "date_joined")

    entries = []
    for rank, row in enumerate(raw_rows, start=1):
        total_xp = int(row["xp_total"] or 0)
        ranking_xp = int(row["ranking_xp"] or 0)
        average_score = round(
            (
                float(row["avg_pronunciation"] or 0)
                + float(row["avg_fluency"] or 0)
                + float(row["avg_grammar"] or 0)
                + float(row["avg_vocabulary"] or 0)
            ) / 4,
            1,
        )
        league = _league_meta(total_xp)

        entries.append(
            {
                "id": str(row["id"]),
                "rank": rank,
                "name": _user_display_name(row),
                "avatar_url": row["avatar_url"] or "",
                "xp": ranking_xp,
                "total_xp": total_xp,
                "level": row["level"] or "beginner",
                "level_number": LEVEL_MAP.get(row["level"], 1),
                "league": league["current_league"],
                "streak_days": int(row["streak_days"] or 0),
                "sessions_count": int(row["activity_sessions"] or row["sessions_count"] or 0),
                "average_score": average_score,
            }
        )

    snapshot = {
        "entries": entries,
        "generated_at": timezone.now(),
        "meta": _scope_meta(scope),
        "top_score": entries[0]["xp"] if entries else 0,
        "best_streak": max((entry["streak_days"] for entry in entries), default=0),
    }
    cache.set(cache_key, snapshot, timeout=300)
    return snapshot


def _with_current_flag(entry, current_user_id):
    return {
        **entry,
        "is_current_user": entry["id"] == current_user_id,
    }


def _leaderboard_payload_for_user(user, scope):
    snapshot = _build_leaderboard_snapshot(scope)
    entries = snapshot["entries"]
    current_user_id = str(user.id)
    current_index = next(
        (index for index, entry in enumerate(entries) if entry["id"] == current_user_id),
        None,
    )

    current_user = _with_current_flag(entries[current_index], current_user_id) if current_index is not None else None
    podium = [_with_current_flag(entry, current_user_id) for entry in entries[:3]]
    leaderboard = [_with_current_flag(entry, current_user_id) for entry in entries[:20]]

    if current_index is None:
        around_me = []
        target_entry = None
        chaser_entry = None
        current_league = _league_meta(user.xp_total)
        user_rank = 0
        current_score = 0
        current_total_xp = user.xp_total
    else:
        start = max(0, current_index - 2)
        end = min(len(entries), current_index + 3)
        around_me = [_with_current_flag(entry, current_user_id) for entry in entries[start:end]]
        target_entry = entries[current_index - 1] if current_index > 0 else None
        chaser_entry = entries[current_index + 1] if current_index + 1 < len(entries) else None
        current_league = _league_meta(current_user["total_xp"])
        user_rank = current_user["rank"]
        current_score = current_user["xp"]
        current_total_xp = current_user["total_xp"]

    total_learners = len(entries)
    if total_learners <= 1 or current_index is None:
        percentile = 100 if total_learners == 1 else 0
    else:
        percentile = round(((total_learners - 1 - current_index) / (total_learners - 1)) * 100)

    summary = {
        **snapshot["meta"],
        "total_learners": total_learners,
        "top_score": snapshot["top_score"],
        "best_streak": snapshot["best_streak"],
        "user_rank": user_rank,
        "user_percentile": percentile,
        "gap_to_target": max(0, (target_entry["xp"] - current_score) if target_entry else 0),
        "lead_over_chaser": max(0, (current_score - chaser_entry["xp"]) if chaser_entry else 0),
        "target_name": target_entry["name"] if target_entry else "",
        "chaser_name": chaser_entry["name"] if chaser_entry else "",
        "current_league": current_league["current_league"],
        "next_league": current_league["next_league"],
        "league_progress": current_league["league_progress"],
        "next_league_target": current_league["next_league_target"],
        "score_to_next_league": current_league["score_to_next_league"],
        "current_score": current_score,
        "current_total_xp": current_total_xp,
        "generated_at": snapshot["generated_at"],
    }

    return {
        "summary": summary,
        "current_user": current_user,
        "podium": podium,
        "leaderboard": leaderboard,
        "around_me": around_me,
    }


class RegisterView(generics.CreateAPIView):
    """POST /api/v1/auth/register/ — Inscription d'un nouvel utilisateur."""

    serializer_class = RegisterSerializer
    permission_classes = [permissions.AllowAny]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "register"

    @extend_schema(
        summary="Inscription",
        description="Crée un nouveau compte T.Speak. Consentement RGPD obligatoire.",
        responses={201: UserProfileSerializer},
    )
    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()

        # Générer les tokens JWT
        refresh = RefreshToken.for_user(user)

        logger.info("Nouvel utilisateur inscrit: %s (langue: %s)", user.email, user.native_language)

        return Response(
            {
                "success": True,
                "message": "Bienvenue sur T.Speak !",
                "user": UserProfileSerializer(user, context={"request": request}).data,
                "tokens": {
                    "access": str(refresh.access_token),
                    "refresh": str(refresh),
                },
            },
            status=status.HTTP_201_CREATED,
        )


class LoginView(TokenObtainPairView):
    """POST /api/v1/auth/login/ — Connexion et obtention des tokens JWT."""

    serializer_class = TSpkTokenObtainSerializer
    permission_classes = [permissions.AllowAny]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "login"

    @extend_schema(summary="Connexion", description="Authentification avec email + mot de passe.")
    def post(self, request, *args, **kwargs):
        response = super().post(request, *args, **kwargs)
        if response.status_code == 200:
            # Enrichir avec les données profil
            try:
                user = User.objects.get(email=request.data.get("email"))
                response.data["user"] = UserProfileSerializer(
                    user, context={"request": request}
                ).data
                response.data["success"] = True
                logger.info("Connexion réussie: %s", user.email)
            except User.DoesNotExist:
                pass
        return response


class LogoutView(generics.GenericAPIView):
    """POST /api/v1/auth/logout/ — Déconnexion et blacklist du refresh token."""

    permission_classes = [permissions.IsAuthenticated]

    @extend_schema(summary="Déconnexion", description="Invalide le refresh token dans Redis.")
    def post(self, request):
        refresh_token = request.data.get("refresh_token")
        if not refresh_token:
            return Response(
                {"success": False, "error": {"message": "refresh_token manquant"}},
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            token = RefreshToken(refresh_token)
            token.blacklist()
            logger.info("Déconnexion: %s", request.user.email)
            return Response({"success": True, "message": "Déconnexion réussie."})
        except TokenError as e:
            return Response(
                {"success": False, "error": {"message": str(e)}},
                status=status.HTTP_400_BAD_REQUEST,
            )


class UserProfileView(generics.RetrieveUpdateAPIView):
    """GET/PATCH /api/v1/auth/me/ — Profil de l'utilisateur connecté."""

    permission_classes = [permissions.IsAuthenticated]

    def get_serializer_class(self):
        if self.request.method in ("PATCH", "PUT"):
            return UserProfileUpdateSerializer
        return UserProfileSerializer

    def get_object(self):
        return self.request.user

    @extend_schema(summary="Profil utilisateur", description="Récupère les données du profil connecté.")
    def retrieve(self, request, *args, **kwargs):
        # Mise en cache du profil (10 minutes)
        cache_key = f"user_profile:{request.user.id}"
        cached = cache.get(cache_key)
        if not cached:
            user = self.get_object()
            cached = UserProfileSerializer(user, context={"request": request}).data
            cache.set(cache_key, cached, timeout=600)

        return Response({"success": True, "data": cached})

    def partial_update(self, request, *args, **kwargs):
        instance = self.get_object()
        serializer = UserProfileUpdateSerializer(instance, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()

        # Invalider le cache
        cache.delete(f"user_profile:{request.user.id}")

        return Response(
            {
                "success": True,
                "message": "Profil mis à jour.",
                "data": UserProfileSerializer(instance, context={"request": request}).data,
            }
        )


class LeaderboardView(generics.ListAPIView):
    """GET /api/v1/auth/leaderboard/ — Classement enrichi hebdo/global."""

    permission_classes = [permissions.IsAuthenticated]

    @extend_schema(
        summary="Classement",
        description="Retourne un leaderboard riche avec podium, rang personnel et vue autour du joueur.",
        parameters=[
            OpenApiParameter(
                name="scope",
                type=str,
                location=OpenApiParameter.QUERY,
                required=False,
                description="`weekly` pour les 7 derniers jours, `global` pour le cumul total.",
            ),
        ],
        responses={200: LeaderboardResponseSerializer},
    )
    def get(self, request):
        scope = request.query_params.get("scope", "weekly").lower()
        if scope not in {"weekly", "global"}:
            scope = "weekly"

        payload = _leaderboard_payload_for_user(request.user, scope)
        serialized = LeaderboardResponseSerializer(payload).data
        return Response(
            {
                "success": True,
                "data": serialized,
                "count": serialized["summary"]["total_learners"],
            }
        )


@api_view(["POST"])
@permission_classes([permissions.IsAuthenticated])
def update_streak(request):
    """POST /api/v1/auth/streak/ — Met à jour le streak quotidien."""
    user = request.user
    old_streak = user.streak_days
    user.update_streak()

    bonus_xp = 0
    if user.streak_days > old_streak:
        # Bonus XP pour maintien du streak
        bonus_xp = min(user.streak_days * 5, 100)
        user.add_xp(bonus_xp)

    return Response(
        {
            "success": True,
            "streak_days": user.streak_days,
            "bonus_xp": bonus_xp,
            "message": f"🔥 Streak de {user.streak_days} jours !",
        }
    )
