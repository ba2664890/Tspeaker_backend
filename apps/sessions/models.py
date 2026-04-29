"""T.Speak — Modèles Sessions Vocales"""

import uuid
from django.db import models
from django.conf import settings


class VocalSession(models.Model):
    """Session d'apprentissage vocal (conversation, exercice ou simulation)."""

    SESSION_TYPES = [
        ("conversation", "Conversation libre"),
        ("simulation", "Simulation professionnelle"),
        ("exercise", "Exercice guidé"),
        ("level_test", "Test de niveau"),
    ]

    STATUS_CHOICES = [
        ("pending", "En attente"),
        ("active", "En cours"),
        ("processing", "Traitement IA"),
        ("completed", "Terminée"),
        ("failed", "Échouée"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="sessions"
    )
    session_type = models.CharField(max_length=50, choices=SESSION_TYPES)
    scenario = models.CharField(max_length=100)
    difficulty = models.CharField(
        max_length=20,
        choices=[("beginner", "Débutant"), ("intermediate", "Intermédiaire"), ("advanced", "Avancé")],
        default="beginner",
    )
    native_language_hint = models.CharField(max_length=50, blank=True, default="")

    # Durée et progression
    duration_sec = models.PositiveIntegerField(default=0)
    exchanges_count = models.PositiveIntegerField(default=0)  # Nb d'aller-retours

    # XP et récompenses
    xp_earned = models.PositiveIntegerField(default=0)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    # Référence à la simulation (si applicable)
    simulation = models.ForeignKey(
        "simulations.Simulation",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="sessions",
    )

    class Meta:
        db_table = "sessions"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "-created_at"]),
            models.Index(fields=["status"]),
            models.Index(fields=["session_type"]),
        ]

    def __str__(self):
        return f"Session {self.session_type} — {self.user.full_name} — {self.created_at:%Y-%m-%d}"


class AudioExchange(models.Model):
    """
    Un échange audio dans une session.
    Représente : question IA → réponse utilisateur → feedback.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(VocalSession, on_delete=models.CASCADE, related_name="exchanges")

    exchange_index = models.PositiveSmallIntegerField()  # Position dans la session

    # Question de l'IA
    ai_question = models.TextField()
    ai_question_audio_url = models.URLField(blank=True, default="")

    # Réponse de l'utilisateur
    user_audio_url = models.URLField(blank=True, default="")
    user_audio_duration_sec = models.FloatField(default=0)
    transcription = models.TextField(blank=True, default="")  # Résultat Whisper

    # Feedback IA
    ai_feedback = models.TextField(blank=True, default="")
    ai_response = models.TextField(blank=True, default="")  # Réponse conversationnelle

    # Scores individuels (0-100)
    pronunciation_score = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    fluency_score = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)

    # Analyse phonémique détaillée (format JSON)
    phoneme_analysis = models.JSONField(default=dict, blank=True)

    # Timing
    processing_time_ms = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    # Tâche Celery associée
    celery_task_id = models.CharField(max_length=50, blank=True, default="")

    class Meta:
        db_table = "audio_exchanges"
        ordering = ["exchange_index"]
        unique_together = ("session", "exchange_index")

    def __str__(self):
        return f"Échange #{self.exchange_index} — Session {self.session_id}"
