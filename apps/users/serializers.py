"""T.Speak — Serializers utilisateurs"""

from django.contrib.auth.password_validation import validate_password
from django.utils import timezone
from rest_framework import serializers
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from .models import User, Badge


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, validators=[validate_password])
    password_confirm = serializers.CharField(write_only=True)
    gdpr_consent = serializers.BooleanField(required=True)

    class Meta:
        model = User
        fields = [
            "email", "first_name", "last_name", "phone", "password", "password_confirm",
            "native_language", "level", "bio", "country", "learning_goal",
            "interests", "age_range", "gdpr_consent",
        ]

    def validate(self, attrs):
        if attrs["password"] != attrs.pop("password_confirm"):
            raise serializers.ValidationError({"password": "Les mots de passe ne correspondent pas."})
        if not attrs.get("gdpr_consent"):
            raise serializers.ValidationError(
                {"gdpr_consent": "Le consentement RGPD est obligatoire pour utiliser T.Speak."}
            )
        return attrs

    def create(self, validated_data):
        user = User.objects.create_user(
            email=validated_data["email"],
            first_name=validated_data["first_name"],
            last_name=validated_data["last_name"],
            password=validated_data["password"],
            phone=validated_data.get("phone"),
            native_language=validated_data.get("native_language", "french"),
            level=validated_data.get("level", "beginner"),
            bio=validated_data.get("bio", ""),
            country=validated_data.get("country", ""),
            learning_goal=validated_data.get("learning_goal", ""),
            interests=validated_data.get("interests", ""),
            age_range=validated_data.get("age_range", ""),
            gdpr_consent=True,
            gdpr_consent_date=timezone.now(),
        )
        return user


class TSpkTokenObtainSerializer(TokenObtainPairSerializer):
    """JWT enrichi avec les données T.Speak."""

    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        token["email"] = user.email
        token["full_name"] = user.full_name
        token["is_premium"] = user.is_premium_active()
        token["level"] = user.level
        return token


class UserProfileSerializer(serializers.ModelSerializer):
    is_premium_active = serializers.SerializerMethodField()
    level_number = serializers.ReadOnlyField()
    xp_for_next_level = serializers.ReadOnlyField()
    badges_count = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = [
            "id", "email", "first_name", "last_name", "full_name", "phone", "bio", "country",
            "learning_goal", "interests", "age_range", "avatar_url",
            "native_language", "level", "level_number",
            "is_premium", "is_premium_active", "premium_until",
            "xp_total", "xp_for_next_level", "streak_days", "sessions_count",
            "avg_pronunciation", "avg_fluency", "avg_grammar", "avg_vocabulary",
            "badges_count", "date_joined",
        ]
        read_only_fields = [
            "id", "email", "xp_total", "streak_days", "sessions_count",
            "avg_pronunciation", "avg_fluency", "avg_grammar", "avg_vocabulary",
            "date_joined",
        ]

    def get_is_premium_active(self, obj):
        return obj.is_premium_active()

    def get_badges_count(self, obj):
        return obj.badges.count()


class UserProfileUpdateSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = [
            "first_name", "last_name", "phone", "avatar_url", 
            "native_language", "bio", "country", "learning_goal", 
            "interests", "age_range"
        ]


class BadgeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Badge
        fields = ["id", "badge_type", "badge_name", "badge_icon", "earned_at"]


class LeaderboardEntrySerializer(serializers.ModelSerializer):
    rank = serializers.SerializerMethodField()
    is_current_user = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = ["id", "first_name", "last_name", "full_name", "avatar_url", "xp_total", "level", "rank", "is_current_user"]

    def get_rank(self, obj):
        return self.context.get("ranks", {}).get(str(obj.id), 0)

    def get_is_current_user(self, obj):
        request = self.context.get("request")
        return request and request.user.id == obj.id
