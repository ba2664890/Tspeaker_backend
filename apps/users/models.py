"""T.Speak — Modèle utilisateur personnalisé"""

import uuid
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.db import models
from django.utils import timezone


class UserManager(BaseUserManager):
    def create_user(self, email, first_name, last_name, password=None, **extra_fields):
        if not email:
            raise ValueError("L'email est obligatoire")
        email = self.normalize_email(email)
        user = self.model(email=email, first_name=first_name, last_name=last_name, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, first_name, last_name, password=None, **extra_fields):
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)
        return self.create_user(email, first_name, last_name, password, **extra_fields)


class User(AbstractBaseUser, PermissionsMixin):
    """
    Modèle utilisateur T.Speak étendu.
    Supporte les langues locales africaines et la gamification.
    """

    LEVEL_CHOICES = [
        ("beginner", "Débutant"),
        ("elementary", "Élémentaire"),
        ("intermediate", "Intermédiaire"),
        ("upper_intermediate", "Intermédiaire avancé"),
        ("advanced", "Avancé"),
    ]

    NATIVE_LANGUAGE_CHOICES = [
        ("wolof", "Wolof"),
        ("pulaar", "Pulaar / Peul"),
        ("bambara", "Bambara"),
        ("dioula", "Dioula"),
        ("serer", "Sérère"),
        ("french", "Français"),
        ("other", "Autre"),
    ]

    # Identifiant
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Authentification
    email = models.EmailField(unique=True, db_index=True)
    phone = models.CharField(max_length=20, unique=True, null=True, blank=True)
    first_name = models.CharField(max_length=50, blank=True)
    last_name = models.CharField(max_length=50, blank=True)
    bio = models.TextField(max_length=500, blank=True)
    country = models.CharField(max_length=100, blank=True)
    avatar_url = models.URLField(blank=True, default="")

    # Profil linguistique et perso
    native_language = models.CharField(
        max_length=50, choices=NATIVE_LANGUAGE_CHOICES, default="french"
    )
    level = models.CharField(max_length=30, choices=LEVEL_CHOICES, default="beginner")
    learning_goal = models.CharField(max_length=100, blank=True)
    interests = models.CharField(max_length=255, blank=True)
    age_range = models.CharField(max_length=20, blank=True)

    # Abonnement
    is_premium = models.BooleanField(default=False)
    premium_until = models.DateTimeField(null=True, blank=True)

    # Gamification
    xp_total = models.PositiveIntegerField(default=0)
    streak_days = models.PositiveIntegerField(default=0)
    streak_last_date = models.DateField(null=True, blank=True)
    sessions_count = models.PositiveIntegerField(default=0)

    # Statistiques cumulées
    avg_pronunciation = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    avg_fluency = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    avg_grammar = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    avg_vocabulary = models.DecimalField(max_digits=5, decimal_places=2, default=0)

    # Permissions Django
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    date_joined = models.DateTimeField(default=timezone.now)
    last_login = models.DateTimeField(null=True, blank=True)

    # Consentement RGPD
    gdpr_consent = models.BooleanField(default=False)
    gdpr_consent_date = models.DateTimeField(null=True, blank=True)

    objects = UserManager()

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["first_name", "last_name"]

    class Meta:
        db_table = "users"
        verbose_name = "Utilisateur"
        verbose_name_plural = "Utilisateurs"
        indexes = [
            models.Index(fields=["email"]),
            models.Index(fields=["xp_total"]),
            models.Index(fields=["streak_days"]),
            models.Index(fields=["native_language"]),
            models.Index(fields=["first_name"]),
            models.Index(fields=["last_name"]),
        ]

    def __str__(self):
        return f"{self.first_name} {self.last_name} <{self.email}>"

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}".strip()

    @property
    def level_number(self):
        """Retourne le niveau sous forme numérique (1-5)."""
        levels = ["beginner", "elementary", "intermediate", "upper_intermediate", "advanced"]
        return levels.index(self.level) + 1

    @property
    def xp_for_next_level(self):
        """XP nécessaire pour le prochain niveau."""
        thresholds = [0, 500, 1500, 3500, 7500, float("inf")]
        current_idx = self.level_number
        return thresholds[current_idx] if current_idx < len(thresholds) else None

    def add_xp(self, amount: int):
        """Ajoute de l'XP et vérifie les passages de niveau."""
        self.xp_total += amount
        self._check_level_up()
        self.save(update_fields=["xp_total", "level"])

    def _check_level_up(self):
        thresholds = {"beginner": 500, "elementary": 1500, "intermediate": 3500, "upper_intermediate": 7500}
        next_levels = {"beginner": "elementary", "elementary": "intermediate",
                       "intermediate": "upper_intermediate", "upper_intermediate": "advanced"}
        if self.level in thresholds and self.xp_total >= thresholds[self.level]:
            self.level = next_levels[self.level]

    def update_streak(self):
        """Met à jour le streak journalier."""
        from datetime import date, timedelta
        today = date.today()
        if self.streak_last_date is None:
            self.streak_days = 1
        elif self.streak_last_date == today - timedelta(days=1):
            self.streak_days += 1
        elif self.streak_last_date < today - timedelta(days=1):
            self.streak_days = 1  # Reset
        self.streak_last_date = today
        self.save(update_fields=["streak_days", "streak_last_date"])

    def is_premium_active(self):
        """Vérifie si l'abonnement premium est actif."""
        if not self.is_premium:
            return False
        if self.premium_until is None:
            return True
        return timezone.now() < self.premium_until


class Badge(models.Model):
    """Badges de gamification gagnés par les utilisateurs."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="badges")
    badge_type = models.CharField(max_length=100)
    badge_name = models.CharField(max_length=100)
    badge_icon = models.CharField(max_length=10, default="🏅")
    earned_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = "badges"
        unique_together = ("user", "badge_type")
        ordering = ["-earned_at"]
        indexes = [models.Index(fields=["user", "badge_type"])]

    def __str__(self):
        return f"{self.user.full_name} — {self.badge_name}"
