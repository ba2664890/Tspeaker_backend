"""T.Speak — Modèle & Vues Scoring"""

import uuid
from django.db import models
from django.conf import settings


class Score(models.Model):
    """Score détaillé d'une session vocale."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.OneToOneField(
        "tspeak_sessions.VocalSession", on_delete=models.CASCADE, related_name="score"
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="scores"
    )

    # Scores par critère (0-100)
    pronunciation = models.DecimalField(max_digits=5, decimal_places=2)
    fluency = models.DecimalField(max_digits=5, decimal_places=2)
    grammar = models.DecimalField(max_digits=5, decimal_places=2)
    vocabulary = models.DecimalField(max_digits=5, decimal_places=2)

    # Score global pondéré (prononciation 30%, fluidité 25%, grammaire 25%, vocab 20%)
    global_score = models.DecimalField(max_digits=5, decimal_places=2)

    # Feedback textuel généré par le LLM
    feedback_text = models.TextField(blank=True, default="")

    # URL audio (S3/MinIO) — supprimé après 24h (RGPD)
    audio_url = models.URLField(max_length=500, blank=True, default="")

    # Méta
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "scores"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "-created_at"]),
            models.Index(fields=["global_score"]),
        ]

    def __str__(self):
        return f"Score {self.global_score}/100 — {self.user.full_name}"

    @classmethod
    def compute_global(cls, pronunciation, fluency, grammar, vocabulary) -> float:
        """Calcule le score global pondéré selon les critères T.Speak."""
        return round(
            pronunciation * 0.30 + fluency * 0.25 + grammar * 0.25 + vocabulary * 0.20,
            2,
        )

    def save(self, *args, **kwargs):
        # Recalcule automatiquement le score global à chaque sauvegarde
        self.global_score = self.compute_global(
            float(self.pronunciation),
            float(self.fluency),
            float(self.grammar),
            float(self.vocabulary),
        )
        super().save(*args, **kwargs)
        # Mettre à jour les moyennes utilisateur
        self._update_user_averages()

    def _update_user_averages(self):
        """Met à jour les stats cumulées de l'utilisateur."""
        from django.db.models import Avg
        user = self.user
        stats = Score.objects.filter(user=user).aggregate(
            avg_p=Avg("pronunciation"),
            avg_f=Avg("fluency"),
            avg_g=Avg("grammar"),
            avg_v=Avg("vocabulary"),
        )
        user.avg_pronunciation = stats["avg_p"] or 0
        user.avg_fluency = stats["avg_f"] or 0
        user.avg_grammar = stats["avg_g"] or 0
        user.avg_vocabulary = stats["avg_v"] or 0
        user.save(update_fields=["avg_pronunciation", "avg_fluency", "avg_grammar", "avg_vocabulary"])
