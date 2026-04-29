"""T.Speak — Modèle & Vues Simulations Professionnelles"""

import uuid
import logging
from django.db import models
from django.core.cache import cache
from rest_framework import generics, permissions, status
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema

logger = logging.getLogger("tspeak.simulations")


class Simulation(models.Model):
    """
    Simulation professionnelle (Pitch investisseur, Entretien, Appel client...).
    Les simulations premium sont réservées aux abonnés.
    """

    CATEGORY_CHOICES = [
        ("pitch", "Pitch Investisseur"),
        ("interview", "Entretien d'embauche"),
        ("client_call", "Appel client"),
        ("crisis", "Gestion de crise"),
        ("negotiation", "Négociation"),
        ("presentation", "Présentation"),
        ("networking", "Networking"),
    ]

    DIFFICULTY_CHOICES = [
        ("beginner", "Débutant"),
        ("intermediate", "Intermédiaire"),
        ("advanced", "Avancé"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    description = models.TextField()
    category = models.CharField(max_length=50, choices=CATEGORY_CHOICES)
    difficulty = models.CharField(max_length=20, choices=DIFFICULTY_CHOICES)
    is_premium = models.BooleanField(default=False)
    language_hint = models.CharField(max_length=50, blank=True, default="")
    duration_min = models.PositiveSmallIntegerField(default=5)
    icon_emoji = models.CharField(max_length=10, default="💼")
    image_url = models.URLField(blank=True, default="")
    rating = models.DecimalField(max_digits=3, decimal_places=1, default=4.5)
    completions_count = models.PositiveIntegerField(default=0)

    # Prompt système pour le LLM (instructions de personnage IA)
    system_prompt = models.TextField(
        help_text="Instructions pour le LLM : personnages, contexte, objectifs de la simulation."
    )

    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "simulations"
        ordering = ["category", "difficulty"]

    def __str__(self):
        return f"{self.icon_emoji} {self.name} ({self.difficulty})"


# ─── Views ───────────────────────────────────────────────────────────────────

from rest_framework import serializers


class SimulationSerializer(serializers.ModelSerializer):
    is_accessible = serializers.SerializerMethodField()

    class Meta:
        model = Simulation
        fields = [
            "id", "name", "description", "category", "difficulty",
            "is_premium", "language_hint", "duration_min",
            "icon_emoji", "image_url", "rating", "completions_count",
            "is_accessible",
        ]

    def get_is_accessible(self, obj):
        request = self.context.get("request")
        if not request or not request.user.is_authenticated:
            return False
        if not obj.is_premium:
            return True
        return request.user.is_premium_active()


class SimulationListView(generics.ListAPIView):
    """GET /api/v1/simulations/ — Liste toutes les simulations disponibles."""

    permission_classes = [permissions.IsAuthenticated]
    serializer_class = SimulationSerializer

    def get_queryset(self):
        qs = Simulation.objects.filter(is_active=True)
        category = self.request.query_params.get("category")
        difficulty = self.request.query_params.get("difficulty")
        if category:
            qs = qs.filter(category=category)
        if difficulty:
            qs = qs.filter(difficulty=difficulty)
        return qs

    def list(self, request, *args, **kwargs):
        cache_key = f"simulations:list:{request.user.is_premium}"
        data = cache.get(cache_key)
        if not data:
            qs = self.get_queryset()
            data = self.get_serializer(qs, many=True, context={"request": request}).data
            cache.set(cache_key, data, timeout=300)
        return Response({"success": True, "data": data, "count": len(data)})


class SimulationDetailView(generics.RetrieveAPIView):
    """GET /api/v1/simulations/{id}/ — Détail d'une simulation."""

    permission_classes = [permissions.IsAuthenticated]
    serializer_class = SimulationSerializer
    queryset = Simulation.objects.filter(is_active=True)

    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()

        # Vérifier accès premium
        if instance.is_premium and not request.user.is_premium_active():
            return Response(
                {
                    "success": False,
                    "error": {
                        "code": "PREMIUM_REQUIRED",
                        "message": "Cette simulation est réservée aux abonnés Premium.",
                    },
                },
                status=status.HTTP_403_FORBIDDEN,
            )

        data = self.get_serializer(instance, context={"request": request}).data
        return Response({"success": True, "data": data})


class SimulationStartView(generics.GenericAPIView):
    """POST /api/v1/simulations/{id}/start/ — Démarre une session de simulation."""

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, pk):
        try:
            simulation = Simulation.objects.get(id=pk, is_active=True)
        except Simulation.DoesNotExist:
            return Response(
                {"success": False, "error": {"message": "Simulation introuvable."}},
                status=status.HTTP_404_NOT_FOUND,
            )

        if simulation.is_premium and not request.user.is_premium_active():
            return Response(
                {"success": False, "error": {"code": "PREMIUM_REQUIRED",
                                              "message": "Abonnement Premium requis."}},
                status=status.HTTP_403_FORBIDDEN,
            )

        # Créer la session
        from apps.sessions.models import VocalSession
        session = VocalSession.objects.create(
            user=request.user,
            session_type="simulation",
            scenario=simulation.name,
            simulation=simulation,
            status="active",
        )

        simulation.completions_count += 1
        simulation.save(update_fields=["completions_count"])

        logger.info(
            "Simulation démarrée: %s — user=%s", simulation.name, request.user.email
        )

        return Response(
            {
                "success": True,
                "session_id": str(session.id),
                "simulation": SimulationSerializer(simulation, context={"request": request}).data,
                "opening_message": _get_opening_message(simulation),
            },
            status=status.HTTP_201_CREATED,
        )


def _get_opening_message(simulation: Simulation) -> str:
    """Génère le message d'ouverture de la simulation."""
    messages = {
        "pitch": "Good morning! I'm Marcus Chen, representing Savanna Capital. I have about 10 minutes. Please, go ahead and tell me about your startup.",
        "interview": "Hello! Please come in. I'm Sarah from HR. We're very excited to meet you today. Can you start by telling us a little about yourself?",
        "client_call": "Hello, this is TechCorp customer support. How can I assist you today?",
        "crisis": "We have a critical situation. Our main product just received major negative press coverage. What's our response strategy?",
        "negotiation": "Thank you for meeting with us. Let's discuss the terms of our potential partnership.",
    }
    return messages.get(simulation.category, "Hello! Let's begin. Please introduce yourself.")
