"""T.Speak — Vues Authentification & Profil utilisateur"""

import logging
from django.core.cache import cache
from rest_framework import generics, status, permissions
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.response import Response
from rest_framework.throttling import AnonRateThrottle
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.exceptions import TokenError
from drf_spectacular.utils import extend_schema, OpenApiResponse

from .models import User, Badge
from .serializers import (
    RegisterSerializer,
    TSpkTokenObtainSerializer,
    UserProfileSerializer,
    UserProfileUpdateSerializer,
    BadgeSerializer,
    LeaderboardEntrySerializer,
)

logger = logging.getLogger("tspeak.auth")


class RegisterView(generics.CreateAPIView):
    """POST /api/v1/auth/register/ — Inscription d'un nouvel utilisateur."""

    serializer_class = RegisterSerializer
    permission_classes = [permissions.AllowAny]
    throttle_classes = [AnonRateThrottle]

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
    """GET /api/v1/auth/leaderboard/ — Classement hebdomadaire."""

    permission_classes = [permissions.IsAuthenticated]

    @extend_schema(summary="Classement", description="Top 50 utilisateurs par XP.")
    def get(self, request):
        cache_key = "leaderboard:weekly"
        data = cache.get(cache_key)

        if not data:
            top_users = User.objects.filter(is_active=True).order_by("-xp_total")[:50]
            ranks = {str(u.id): idx + 1 for idx, u in enumerate(top_users)}
            data = LeaderboardEntrySerializer(
                top_users, many=True, context={"request": request, "ranks": ranks}
            ).data
            cache.set(cache_key, data, timeout=300)  # Cache 5 min

        return Response({"success": True, "data": data, "count": len(data)})


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
