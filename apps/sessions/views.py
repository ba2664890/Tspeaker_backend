"""
T.Speak — Vues Sessions Vocales
Pipeline complet : upload audio → Celery → Whisper → Wav2Vec → LLM → score
"""

import logging
import os
import uuid
from datetime import timedelta

from django.core.cache import cache
from django.utils import timezone
from rest_framework import generics, status, permissions
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.throttling import UserRateThrottle
from drf_spectacular.utils import extend_schema

from .models import VocalSession, AudioExchange
from .serializers import (
    SessionStartSerializer,
    SessionDetailSerializer,
    SessionHistorySerializer,
    AudioUploadSerializer,
)
from .tasks import process_audio_exchange
from .validators import validate_audio_file
from apps.users.models import User

logger = logging.getLogger("tspeak.sessions")


class AudioUploadThrottle(UserRateThrottle):
    scope = "audio_upload"


class SessionStartView(generics.CreateAPIView):
    """POST /api/v1/sessions/start/ — Démarre une nouvelle session vocale."""

    permission_classes = [permissions.IsAuthenticated]
    serializer_class = SessionStartSerializer

    def create(self, request, *args, **kwargs):
        user = request.user

        # Vérifier la limite de sessions (utilisateurs gratuits : 5/jour)
        if not user.is_premium_active():
            today_count = VocalSession.objects.filter(
                user=user,
                created_at__date=timezone.now().date(),
                status="completed",
            ).count()
            if today_count >= 5:
                return Response(
                    {
                        "success": False,
                        "error": {
                            "code": "SESSION_LIMIT_REACHED",
                            "message": "Limite de 5 sessions/jour atteinte. Passez Premium pour des sessions illimitées !",
                        },
                    },
                    status=status.HTTP_403_FORBIDDEN,
                )

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        session = VocalSession.objects.create(
            user=user,
            session_type=serializer.validated_data["session_type"],
            scenario=serializer.validated_data["scenario"],
            difficulty=serializer.validated_data.get("difficulty", user.level),
            native_language_hint=user.native_language,
            simulation_id=serializer.validated_data.get("simulation_id"),
            status="active",
            started_at=timezone.now(),
        )

        logger.info(
            "Session démarrée: %s (%s) — user=%s",
            session.scenario, session.session_type, user.email,
        )

        return Response(
            {
                "success": True,
                "session_id": str(session.id),
                "message": "Session démarrée. Bonne pratique !",
                "first_question": _generate_first_question(session),
            },
            status=status.HTTP_201_CREATED,
        )


class AudioUploadView(generics.GenericAPIView):
    """
    POST /api/v1/sessions/{id}/audio/
    Upload d'un fichier audio pour traitement par l'IA.
    Pipeline : WAV → Celery → Whisper + Wav2Vec → LLM → Score
    """

    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]
    throttle_classes = [AudioUploadThrottle]

    @extend_schema(
        summary="Upload audio",
        description="Envoie un fichier audio pour transcription et scoring. Retourne un task_id Celery.",
    )
    def post(self, request, session_id):
        # Récupérer la session
        try:
            session = VocalSession.objects.get(id=session_id, user=request.user)
        except VocalSession.DoesNotExist:
            return Response(
                {"success": False, "error": {"message": "Session introuvable."}},
                status=status.HTTP_404_NOT_FOUND,
            )

        if session.status != "active":
            return Response(
                {"success": False, "error": {"message": f"Session en état '{session.status}', pas active."}},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Validation du fichier audio
        audio_file = request.FILES.get("audio")
        if not audio_file:
            return Response(
                {"success": False, "error": {"message": "Fichier audio manquant (champ 'audio')."}},
                status=status.HTTP_400_BAD_REQUEST,
            )

        validation_error = validate_audio_file(audio_file)
        if validation_error:
            return Response(
                {"success": False, "error": {"message": validation_error}},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Sauvegarder le fichier temporairement
        temp_path = _save_temp_audio(audio_file, session_id)

        # Créer l'échange
        exchange = AudioExchange.objects.create(
            session=session,
            exchange_index=session.exchanges_count,
            ai_question=request.data.get("question", ""),
            user_audio_duration_sec=float(request.data.get("duration_sec", 0)),
        )

        # Envoyer à Celery pour traitement asynchrone
        task = process_audio_exchange.apply_async(
            args=[str(exchange.id), temp_path, str(request.user.native_language)],
            queue="audio",
            countdown=0,
        )

        exchange.celery_task_id = task.id
        exchange.save(update_fields=["celery_task_id"])

        session.exchanges_count += 1
        session.status = "processing"
        session.save(update_fields=["exchanges_count", "status"])

        logger.info(
            "Audio envoyé en traitement: exchange=%s task=%s", exchange.id, task.id
        )

        return Response(
            {
                "success": True,
                "exchange_id": str(exchange.id),
                "task_id": task.id,
                "message": "Audio en cours de traitement. Résultat disponible dans ~2-3 secondes.",
            },
            status=status.HTTP_202_ACCEPTED,
        )


class AudioResultView(generics.RetrieveAPIView):
    """GET /api/v1/sessions/exchanges/{exchange_id}/result/ — Récupère le résultat d'un échange."""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, exchange_id):
        # Vérifier le cache Redis d'abord
        cache_key = f"exchange_result:{exchange_id}"
        cached = cache.get(cache_key)
        if cached:
            return Response({"success": True, "status": "completed", "data": cached})

        try:
            exchange = AudioExchange.objects.select_related("session").get(
                id=exchange_id, session__user=request.user
            )
        except AudioExchange.DoesNotExist:
            return Response(
                {"success": False, "error": {"message": "Échange introuvable."}},
                status=status.HTTP_404_NOT_FOUND,
            )

        if not exchange.transcription:
            # Vérifier le statut Celery
            from celery.result import AsyncResult
            result = AsyncResult(exchange.celery_task_id)
            return Response(
                {
                    "success": True,
                    "status": result.state,
                    "message": "Traitement en cours...",
                    "task_id": exchange.celery_task_id,
                }
            )

        data = {
            "exchange_id": str(exchange.id),
            "transcription": exchange.transcription,
            "ai_feedback": exchange.ai_feedback,
            "ai_response": exchange.ai_response,
            "pronunciation_score": float(exchange.pronunciation_score or 0),
            "fluency_score": float(exchange.fluency_score or 0),
            "phoneme_analysis": exchange.phoneme_analysis,
            "processing_time_ms": exchange.processing_time_ms,
        }
        # Mettre en cache 10 minutes
        cache.set(cache_key, data, timeout=600)
        return Response({"success": True, "status": "completed", "data": data})


class SessionDetailView(generics.RetrieveAPIView):
    """GET /api/v1/sessions/{id}/ — Détails d'une session."""

    permission_classes = [permissions.IsAuthenticated]
    serializer_class = SessionDetailSerializer

    def get_queryset(self):
        return VocalSession.objects.filter(user=self.request.user).prefetch_related("exchanges")

    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        data = self.get_serializer(instance).data
        return Response({"success": True, "data": data})


class SessionHistoryView(generics.ListAPIView):
    """GET /api/v1/sessions/history/ — Historique des sessions."""

    permission_classes = [permissions.IsAuthenticated]
    serializer_class = SessionHistorySerializer

    def get_queryset(self):
        user = self.request.user
        qs = VocalSession.objects.filter(user=user, status="completed").order_by("-created_at")

        # Utilisateurs gratuits : 7 jours max
        if not user.is_premium_active():
            cutoff = timezone.now() - timedelta(days=7)
            qs = qs.filter(created_at__gte=cutoff)

        return qs

    def list(self, request, *args, **kwargs):
        qs = self.get_queryset()
        data = self.get_serializer(qs, many=True).data
        return Response({"success": True, "data": data, "count": len(data)})


@api_view(["POST"])
@permission_classes([permissions.IsAuthenticated])
def end_session(request, session_id):
    """POST /api/v1/sessions/{id}/end/ — Termine une session et calcule le score final."""
    try:
        session = VocalSession.objects.get(id=session_id, user=request.user)
    except VocalSession.DoesNotExist:
        return Response(
            {"success": False, "error": {"message": "Session introuvable."}},
            status=status.HTTP_404_NOT_FOUND,
        )

    if session.status not in ("active", "processing"):
        return Response(
            {"success": False, "error": {"message": "Session déjà terminée."}},
            status=status.HTTP_400_BAD_REQUEST,
        )

    duration = int(request.data.get("duration_sec", 0))
    session.duration_sec = duration
    session.status = "completed"
    session.completed_at = timezone.now()

    # Calculer XP gagnés
    from core.settings import XP_PER_SESSION
    session.xp_earned = XP_PER_SESSION
    session.save()

    # Mettre à jour les stats utilisateur
    user = request.user
    user.sessions_count += 1
    user.add_xp(session.xp_earned)
    user.update_streak()
    user.save(update_fields=["sessions_count"])

    # Vérifier les badges
    from apps.progress.tasks import check_and_award_badges
    check_and_award_badges.delay(str(user.id))

    return Response(
        {
            "success": True,
            "xp_earned": session.xp_earned,
            "streak_days": user.streak_days,
            "message": f"✅ Session terminée ! +{session.xp_earned} XP",
        }
    )


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _save_temp_audio(audio_file, session_id: str) -> str:
    """Sauvegarde temporaire du fichier audio pour traitement Celery."""
    import tempfile
    ext = audio_file.name.rsplit(".", 1)[-1].lower()
    temp_dir = "/tmp/tspeak_audio"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{session_id}_{uuid.uuid4().hex[:8]}.{ext}")
    with open(temp_path, "wb") as f:
        for chunk in audio_file.chunks():
            f.write(chunk)
    return temp_path


def _generate_first_question(session: VocalSession) -> str:
    """Génère la première question IA selon le scénario."""
    questions = {
        "conversation": "Tell me about yourself. What do you do for a living?",
        "simulation": "Good morning! I'd like to start by asking you to introduce your project.",
        "exercise": "Let's begin with a simple exercise. Please repeat after me: 'The weather is beautiful today.'",
        "level_test": "Hello! Let's assess your English level. Can you tell me about your daily routine?",
    }
    return questions.get(session.session_type, "Hello! How are you today?")
