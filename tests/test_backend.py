"""
T.Speak — Tests Backend Complets
=================================

Tests unitaires, d'intégration et du pipeline IA.
Objectif couverture : > 80%

Lancer :
    pytest tests/ -v --cov=apps --cov-report=html
    pytest tests/test_ai_pipeline.py -v -k "whisper"
"""

import json
import os
import tempfile
from io import BytesIO
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient
from rest_framework_simplejwt.tokens import RefreshToken

User = get_user_model()


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def api_client():
    return APIClient()


@pytest.fixture
def user(db):
    return User.objects.create_user(
        email="test@tspeak.africa",
        full_name="Amadou Diallo",
        password="TestPass123!",
        native_language="wolof",
        level="beginner",
        gdpr_consent=True,
    )


@pytest.fixture
def premium_user(db):
    return User.objects.create_user(
        email="premium@tspeak.africa",
        full_name="Fatou Sow",
        password="TestPass123!",
        native_language="wolof",
        is_premium=True,
        gdpr_consent=True,
    )


@pytest.fixture
def auth_client(api_client, user):
    refresh = RefreshToken.for_user(user)
    api_client.credentials(HTTP_AUTHORIZATION=f"Bearer {str(refresh.access_token)}")
    return api_client


@pytest.fixture
def premium_client(api_client, premium_user):
    refresh = RefreshToken.for_user(premium_user)
    api_client.credentials(HTTP_AUTHORIZATION=f"Bearer {str(refresh.access_token)}")
    return api_client


@pytest.fixture
def fake_wav_audio():
    """Crée un fichier WAV factice de 1 seconde."""
    import wave
    import struct
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        # 1 seconde de silence (16000 frames à 16 bits)
        frames = struct.pack("<" + "h" * 16000, *([0] * 16000))
        wf.writeframes(frames)
    buffer.seek(0)
    buffer.name = "test_audio.wav"
    return buffer


# ─── Tests Authentification ───────────────────────────────────────────────────

@pytest.mark.django_db
class TestAuthentication:

    def test_register_success(self, api_client):
        url = reverse("auth-register")
        data = {
            "email": "new@tspeak.africa",
            "full_name": "Cheikh Ba",
            "password": "SecurePass123!",
            "password_confirm": "SecurePass123!",
            "native_language": "wolof",
            "gdpr_consent": True,
        }
        response = api_client.post(url, data, format="json")
        assert response.status_code == status.HTTP_201_CREATED
        assert response.data["success"] is True
        assert "tokens" in response.data
        assert "access" in response.data["tokens"]
        assert User.objects.filter(email="new@tspeak.africa").exists()

    def test_register_no_gdpr_consent_fails(self, api_client):
        url = reverse("auth-register")
        data = {
            "email": "nogdpr@test.com",
            "full_name": "Test User",
            "password": "SecurePass123!",
            "password_confirm": "SecurePass123!",
            "gdpr_consent": False,
        }
        response = api_client.post(url, data, format="json")
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_register_password_mismatch(self, api_client):
        url = reverse("auth-register")
        data = {
            "email": "test@test.com",
            "full_name": "Test",
            "password": "Pass123!",
            "password_confirm": "DifferentPass!",
            "gdpr_consent": True,
        }
        response = api_client.post(url, data, format="json")
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_login_success(self, api_client, user):
        url = reverse("auth-login")
        response = api_client.post(url, {"email": user.email, "password": "TestPass123!"})
        assert response.status_code == status.HTTP_200_OK
        assert "access" in response.data

    def test_login_wrong_password(self, api_client, user):
        url = reverse("auth-login")
        response = api_client.post(url, {"email": user.email, "password": "WrongPass!"})
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_get_profile_authenticated(self, auth_client, user):
        url = reverse("auth-me")
        response = auth_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        assert response.data["data"]["email"] == user.email

    def test_get_profile_unauthenticated(self, api_client):
        url = reverse("auth-me")
        response = api_client.get(url)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_update_profile(self, auth_client):
        url = reverse("auth-me")
        response = auth_client.patch(url, {"full_name": "Nouveau Nom"}, format="json")
        assert response.status_code == status.HTTP_200_OK
        assert response.data["data"]["full_name"] == "Nouveau Nom"

    def test_logout(self, auth_client, user):
        refresh = str(RefreshToken.for_user(user))
        response = auth_client.post(reverse("auth-logout"), {"refresh_token": refresh})
        assert response.status_code == status.HTTP_200_OK

    def test_streak_update(self, auth_client, user):
        url = reverse("auth-streak")
        response = auth_client.post(url)
        assert response.status_code == status.HTTP_200_OK
        assert "streak_days" in response.data
        assert response.data["streak_days"] == 1


# ─── Tests Sessions ───────────────────────────────────────────────────────────

@pytest.mark.django_db
class TestSessions:

    def test_start_session(self, auth_client):
        url = reverse("session-start")
        data = {
            "session_type": "conversation",
            "scenario": "daily_life",
            "difficulty": "beginner",
        }
        response = auth_client.post(url, data, format="json")
        assert response.status_code == status.HTTP_201_CREATED
        assert "session_id" in response.data
        assert "first_question" in response.data

    def test_free_user_session_limit(self, auth_client, user):
        """Utilisateur gratuit limité à 5 sessions par jour."""
        from apps.sessions.models import VocalSession
        from django.utils import timezone

        # Créer 5 sessions complètes
        for i in range(5):
            VocalSession.objects.create(
                user=user,
                session_type="conversation",
                scenario=f"test_{i}",
                status="completed",
                duration_sec=60,
            )

        url = reverse("session-start")
        response = auth_client.post(url, {
            "session_type": "conversation",
            "scenario": "daily_life",
        }, format="json")

        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert "SESSION_LIMIT_REACHED" in str(response.data)

    def test_premium_user_no_session_limit(self, premium_client, premium_user):
        """Utilisateur premium : sessions illimitées."""
        from apps.sessions.models import VocalSession

        # Créer 10 sessions (au-delà de la limite gratuite)
        for i in range(10):
            VocalSession.objects.create(
                user=premium_user,
                session_type="conversation",
                scenario=f"test_{i}",
                status="completed",
                duration_sec=60,
            )

        url = reverse("session-start")
        response = premium_client.post(url, {
            "session_type": "conversation",
            "scenario": "daily_life",
        }, format="json")

        assert response.status_code == status.HTTP_201_CREATED

    def test_session_history_free_user_limited_to_7_days(self, auth_client, user):
        """Historique limité à 7 jours pour utilisateurs gratuits."""
        from apps.sessions.models import VocalSession
        from django.utils import timezone
        from datetime import timedelta

        # Session il y a 10 jours (hors limite)
        old_session = VocalSession.objects.create(
            user=user,
            session_type="conversation",
            scenario="old",
            status="completed",
            duration_sec=60,
        )
        old_session.created_at = timezone.now() - timedelta(days=10)
        old_session.save(update_fields=["created_at"])

        # Session récente
        VocalSession.objects.create(
            user=user,
            session_type="conversation",
            scenario="recent",
            status="completed",
            duration_sec=60,
        )

        url = reverse("session-history")
        response = auth_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        assert response.data["count"] == 1  # Seulement la session récente


# ─── Tests Scoring ────────────────────────────────────────────────────────────

@pytest.mark.django_db
class TestScoring:

    def test_global_score_computation(self):
        from apps.scoring.models import Score
        global_score = Score.compute_global(
            pronunciation=80.0,
            fluency=75.0,
            grammar=70.0,
            vocabulary=65.0,
        )
        expected = 80 * 0.30 + 75 * 0.25 + 70 * 0.25 + 65 * 0.20
        assert abs(global_score - expected) < 0.01

    def test_score_saves_and_updates_user_averages(self, user):
        from apps.sessions.models import VocalSession
        from apps.scoring.models import Score

        session = VocalSession.objects.create(
            user=user,
            session_type="conversation",
            scenario="test",
            status="completed",
            duration_sec=60,
        )
        score = Score.objects.create(
            session=session,
            user=user,
            pronunciation=80.0,
            fluency=75.0,
            grammar=70.0,
            vocabulary=65.0,
        )
        user.refresh_from_db()
        assert float(user.avg_pronunciation) == 80.0


# ─── Tests IA Pipeline ────────────────────────────────────────────────────────

class TestWhisperTranscriber:
    """Tests du module Whisper (sans chargement du vrai modèle)."""

    @patch("ai.whisper_asr.transcriber.whisper")
    def test_transcribe_returns_expected_format(self, mock_whisper):
        from ai.whisper_asr.transcriber import WhisperTranscriber

        # Mock du modèle Whisper
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "Hello, my name is Amadou",
            "language": "en",
            "segments": [
                {
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 0.95},
                        {"word": "my", "start": 0.5, "end": 0.7, "probability": 0.90},
                    ],
                    "no_speech_prob": 0.02,
                }
            ],
        }
        mock_whisper.load_model.return_value = mock_model

        transcriber = WhisperTranscriber(model_name="tiny")
        transcriber._model = mock_model

        result = transcriber.transcribe("/fake/path.wav")

        assert "text" in result
        assert "words" in result
        assert "avg_confidence" in result
        assert result["text"] == "Hello, my name is Amadou"
        assert len(result["words"]) == 2

    def test_compute_wer(self):
        from ai.whisper_asr.transcriber import WhisperTranscriber
        transcriber = WhisperTranscriber.__new__(WhisperTranscriber)
        wer = transcriber.compute_wer(
            reference="hello my name is amadou",
            hypothesis="hello my name is amadou",
        )
        assert wer == 0.0

        wer_imperfect = transcriber.compute_wer(
            reference="hello my name is amadou",
            hypothesis="hello my name is amadu",
        )
        assert wer_imperfect > 0.0


class TestGrammarAnalyzer:

    def test_correct_sentence_high_score(self):
        from ai.wav2vec_scoring.nlp_analyzer import GrammarAnalyzer
        analyzer = GrammarAnalyzer()
        result = analyzer.analyze("She goes to the market every morning.")
        assert result["grammar_score"] >= 70

    def test_sentence_complexity_detection(self):
        from ai.wav2vec_scoring.nlp_analyzer import GrammarAnalyzer
        analyzer = GrammarAnalyzer()
        complex_text = "Although it was raining, we decided to go to the market because we needed food."
        simple_text = "I go market."
        complex_result = analyzer.analyze(complex_text)
        simple_result = analyzer.analyze(simple_text)
        assert complex_result["sentence_complexity"] > simple_result["sentence_complexity"]


class TestVocabularyAnalyzer:

    def test_advanced_vocabulary_high_score(self):
        from ai.wav2vec_scoring.nlp_analyzer import VocabularyAnalyzer
        analyzer = VocabularyAnalyzer()
        text = "We need to leverage innovative strategies to optimize our sustainable impact."
        result = analyzer.analyze(text)
        assert result["vocabulary_score"] > 70
        assert result["cefr_level"] in ("B2", "C1", "C2")

    def test_basic_vocabulary_low_score(self):
        from ai.wav2vec_scoring.nlp_analyzer import VocabularyAnalyzer
        analyzer = VocabularyAnalyzer()
        text = "I go to the market. I buy food. I eat food."
        result = analyzer.analyze(text)
        assert result["vocabulary_score"] < 65
        assert result["cefr_level"] in ("A1", "A2", "B1")

    def test_type_token_ratio(self):
        from ai.wav2vec_scoring.nlp_analyzer import VocabularyAnalyzer
        analyzer = VocabularyAnalyzer()
        diverse_text = "The innovative entrepreneur developed sustainable solutions for African markets."
        repetitive_text = "I go go go market market market market market market."
        diverse_result = analyzer.analyze(diverse_text)
        repetitive_result = analyzer.analyze(repetitive_text)
        assert diverse_result["type_token_ratio"] > repetitive_result["type_token_ratio"]


class TestFluencyScore:

    def test_ideal_speech_rate_high_score(self):
        from apps.sessions.tasks import _compute_fluency_score
        # 150 wpm = idéal
        text = " ".join(["word"] * 75)  # 75 mots
        score = _compute_fluency_score(text, duration_sec=30.0)  # 150 wpm
        assert score >= 85.0

    def test_too_slow_lower_score(self):
        from apps.sessions.tasks import _compute_fluency_score
        text = " ".join(["word"] * 20)  # 20 mots en 30 sec = 40 wpm (trop lent)
        score = _compute_fluency_score(text, duration_sec=30.0)
        assert score < 65.0

    def test_very_short_response_penalty(self):
        from apps.sessions.tasks import _compute_fluency_score
        score = _compute_fluency_score("Yes", duration_sec=1.0)
        assert score < 60.0


# ─── Tests Performance & Cache ────────────────────────────────────────────────

@pytest.mark.django_db
class TestCaching:

    def test_leaderboard_cached(self, auth_client):
        cache.clear()
        url = reverse("auth-leaderboard")

        # Premier appel — calcul
        response1 = auth_client.get(url)
        assert response1.status_code == status.HTTP_200_OK

        # Deuxième appel — doit venir du cache
        response2 = auth_client.get(url)
        assert response2.status_code == status.HTTP_200_OK
        assert cache.get("leaderboard:weekly") is not None

    def test_profile_cache_invalidated_on_update(self, auth_client, user):
        profile_url = reverse("auth-me")

        # Remplir le cache
        auth_client.get(profile_url)
        assert cache.get(f"user_profile:{user.id}") is not None

        # Mise à jour → doit invalider le cache
        auth_client.patch(profile_url, {"full_name": "Nouveau Nom"}, format="json")
        assert cache.get(f"user_profile:{user.id}") is None


# ─── Tests Simulations ────────────────────────────────────────────────────────

@pytest.mark.django_db
class TestSimulations:

    def test_list_simulations(self, auth_client):
        from apps.simulations.models import Simulation
        Simulation.objects.create(
            name="Pitch Investisseur",
            description="Simulation de pitch",
            category="pitch",
            difficulty="intermediate",
            is_premium=False,
            duration_min=10,
            system_prompt="You are an investor.",
        )
        url = reverse("simulation-list")
        response = auth_client.get(url)
        assert response.status_code == status.HTTP_200_OK
        assert response.data["count"] >= 1

    def test_premium_simulation_blocked_for_free_user(self, auth_client):
        from apps.simulations.models import Simulation
        sim = Simulation.objects.create(
            name="Pitch Avancé",
            description="Premium only",
            category="pitch",
            difficulty="advanced",
            is_premium=True,
            duration_min=15,
            system_prompt="You are an investor.",
        )
        url = reverse("simulation-detail", kwargs={"pk": sim.id})
        response = auth_client.get(url)
        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_premium_simulation_accessible_for_premium(self, premium_client):
        from apps.simulations.models import Simulation
        sim = Simulation.objects.create(
            name="Pitch Avancé",
            description="Premium only",
            category="pitch",
            difficulty="advanced",
            is_premium=True,
            duration_min=15,
            system_prompt="You are an investor.",
        )
        url = reverse("simulation-detail", kwargs={"pk": sim.id})
        response = premium_client.get(url)
        assert response.status_code == status.HTTP_200_OK


# ─── Configuration Pytest ─────────────────────────────────────────────────────

# pytest.ini ou conftest.py settings
@pytest.fixture(autouse=True)
def clear_cache():
    """Vider le cache avant chaque test."""
    cache.clear()
    yield
    cache.clear()
