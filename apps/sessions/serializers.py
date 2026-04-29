"""T.Speak — Serializers Sessions"""
from rest_framework import serializers
from .models import VocalSession, AudioExchange


class SessionStartSerializer(serializers.Serializer):
    session_type = serializers.ChoiceField(choices=["conversation", "simulation", "exercise", "level_test"])
    scenario = serializers.CharField(max_length=100, default="daily_life")
    difficulty = serializers.ChoiceField(choices=["beginner", "intermediate", "advanced"], required=False)
    simulation_id = serializers.UUIDField(required=False, allow_null=True)


class AudioExchangeSerializer(serializers.ModelSerializer):
    class Meta:
        model = AudioExchange
        fields = [
            "id", "exchange_index", "ai_question",
            "transcription", "ai_feedback", "ai_response",
            "pronunciation_score", "fluency_score",
            "phoneme_analysis", "processing_time_ms", "created_at",
        ]


class SessionDetailSerializer(serializers.ModelSerializer):
    exchanges = AudioExchangeSerializer(many=True, read_only=True)

    class Meta:
        model = VocalSession
        fields = [
            "id", "session_type", "scenario", "difficulty",
            "duration_sec", "exchanges_count", "xp_earned",
            "status", "created_at", "completed_at", "exchanges",
        ]


class SessionHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = VocalSession
        fields = [
            "id", "session_type", "scenario", "difficulty",
            "duration_sec", "exchanges_count", "xp_earned",
            "status", "created_at", "completed_at",
        ]


class AudioUploadSerializer(serializers.Serializer):
    audio = serializers.FileField()
    question = serializers.CharField(required=False, default="")
    duration_sec = serializers.FloatField(required=False, default=0)
