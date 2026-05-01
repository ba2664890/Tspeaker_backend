"""
T.Speak — Tâches Celery : Pipeline de traitement audio
Whisper (transcription) → Wav2Vec (scoring phonétique) → LLM (feedback)
"""

import logging
import os
import time

from celery import shared_task
from django.core.cache import cache

logger = logging.getLogger("tspeak.ai")


@shared_task(
    bind=True,
    name="sessions.process_audio_exchange",
    max_retries=3,
    default_retry_delay=5,
    queue="audio",
    acks_late=True,
    reject_on_worker_lost=True,
)
def process_audio_exchange(self, exchange_id: str, audio_path: str, native_language: str):
    """
    Pipeline principal de traitement audio.

    1. Conversion + normalisation audio (FFmpeg)
    2. Transcription Whisper
    3. Scoring phonétique Wav2Vec
    4. Analyse grammaticale NLP
    5. Génération feedback LLM
    6. Calcul score global
    7. Sauvegarde PostgreSQL + cache Redis
    """
    try:
        return process_audio_exchange_now(exchange_id, audio_path, native_language)

    except Exception as exc:
        logger.error("❌ Erreur traitement audio: exchange=%s — %s", exchange_id, exc, exc_info=True)
        try:
            raise self.retry(exc=exc)
        except Exception:
            # Marquer la session comme échouée après 3 tentatives
            try:
                from apps.sessions.models import AudioExchange
                exchange = AudioExchange.objects.get(id=exchange_id)
                exchange.ai_feedback = "Une erreur s'est produite. Veuillez réessayer."
                exchange.session.status = "active"
                exchange.save(update_fields=["ai_feedback"])
                exchange.session.save(update_fields=["status"])
            except Exception:
                logger.exception("Impossible de marquer l'échange %s en erreur", exchange_id)
            return {"status": "error", "message": str(exc)}


def process_audio_exchange_now(exchange_id: str, audio_path: str, native_language: str):
    """Exécute le pipeline audio immédiatement, sans passer par un worker Celery."""
    from apps.sessions.models import AudioExchange

    start_time = time.monotonic()
    logger.info("🎙️ Traitement audio démarré: exchange=%s", exchange_id)

    try:
        exchange = AudioExchange.objects.select_related("session__user").get(id=exchange_id)
        if exchange.transcription:
            logger.info("Échange déjà traité (skip): %s", exchange_id)
            return {"status": "success", "already_processed": True}
    except AudioExchange.DoesNotExist:
        logger.error("Échange introuvable: %s", exchange_id)
        return {"status": "error", "message": "Exchange not found"}

    if not os.path.exists(audio_path):
        # Si le fichier n'existe pas, c'est peut-être qu'un autre worker (ou le fallback)
        # l'a déjà traité et supprimé. On vérifie la transcription une dernière fois.
        exchange.refresh_from_db()
        if exchange.transcription:
            return {"status": "success", "already_processed": True}
        logger.error("Fichier audio introuvable (et non traité): %s", audio_path)
        return {"status": "error", "message": f"Audio file not found: {audio_path}"}

    wav_path = None
    processing_succeeded = False
    try:
        # ── Étape 1 : Conversion audio ─────────────────────────────────────
        wav_path = _convert_to_wav(audio_path)

        # ── Étape 2 : Transcription Whisper ───────────────────────────────
        from ai.whisper_asr.transcriber import get_transcriber
        transcriber = get_transcriber()
        transcription_result = transcriber.transcribe(wav_path, language="en")
        transcription = transcription_result["text"].strip()
        logger.info("Transcription: '%s...'", transcription[:50])

        # ── Étape 3 : Scoring Wav2Vec ──────────────────────────────────────
        from ai.wav2vec_scoring.scorer import get_scorer
        scorer = get_scorer()
        phoneme_analysis = scorer.score_pronunciation(
            wav_path,
            reference_text=exchange.ai_question,
            user_text=transcription,
        )
        pronunciation_score = phoneme_analysis["pronunciation_score"]
        fluency_score = _compute_fluency_score(
            transcription,
            duration_sec=exchange.user_audio_duration_sec,
        )

        # ── Étape 4 : Analyse grammaire + vocabulaire ────────────────────
        from ai.wav2vec_scoring.nlp_analyzer import GrammarAnalyzer, VocabularyAnalyzer
        grammar_analysis = GrammarAnalyzer().analyze(transcription)
        vocabulary_analysis = VocabularyAnalyzer().analyze(transcription)
        grammar_score = grammar_analysis["grammar_score"]
        vocabulary_score = vocabulary_analysis["vocabulary_score"]

        # ── Étape 5 : Génération feedback LLM ─────────────────────────────
        from ai.llm_conversation.generator import get_generator
        generator = get_generator()
        session = exchange.session
        history = _get_session_history(session)

        llm_response = generator.generate_feedback(
            user_transcription=transcription,
            ai_question=exchange.ai_question,
            pronunciation_score=pronunciation_score,
            fluency_score=fluency_score,
            native_language=native_language,
            session_type=session.session_type,
            history=history,
        )

        # ── Étape 6 : Sauvegarde ──────────────────────────────────────────
        processing_ms = int((time.monotonic() - start_time) * 1000)

        exchange.transcription = transcription
        exchange.pronunciation_score = pronunciation_score
        exchange.fluency_score = fluency_score
        exchange.phoneme_analysis = phoneme_analysis
        exchange.ai_feedback = llm_response["feedback"]
        exchange.ai_response = llm_response["next_question"]
        exchange.processing_time_ms = processing_ms
        exchange.save()

        from apps.scoring.models import Score
        score, _ = Score.objects.update_or_create(
            session=session,
            defaults={
                "user": session.user,
                "pronunciation": pronunciation_score,
                "fluency": fluency_score,
                "grammar": grammar_score,
                "vocabulary": vocabulary_score,
                "feedback_text": llm_response["feedback"],
            },
        )

        session.status = "active"
        session.save(update_fields=["status"])

        result_data = {
            "exchange_id": exchange_id,
            "transcription": transcription,
            "pronunciation_score": pronunciation_score,
            "fluency_score": fluency_score,
            "grammar_score": grammar_score,
            "vocabulary_score": vocabulary_score,
            "global_score": float(score.global_score),
            "ai_feedback": llm_response["feedback"],
            "ai_response": llm_response["next_question"],
            "phoneme_analysis": phoneme_analysis,
            "grammar_analysis": grammar_analysis,
            "vocabulary_analysis": vocabulary_analysis,
            "processing_time_ms": processing_ms,
        }
        cache.set(f"exchange_result:{exchange_id}", result_data, timeout=600)

        logger.info(
            "✅ Traitement terminé: exchange=%s — %dms — prononciation=%.1f",
            exchange_id, processing_ms, pronunciation_score,
        )
        processing_succeeded = True
        return result_data

    finally:
        if processing_succeeded and wav_path:
            _cleanup_audio(wav_path)
        if processing_succeeded and audio_path != wav_path:
            _cleanup_audio(audio_path)


@shared_task(name="sessions.cleanup_audio_files")
def cleanup_audio_files():
    """Nettoyage RGPD : supprime les fichiers audio > 24h."""
    import glob
    import time

    audio_dir = "/tmp/tspeak_audio"
    if not os.path.exists(audio_dir):
        return

    count = 0
    cutoff = time.time() - (24 * 3600)
    for filepath in glob.glob(os.path.join(audio_dir, "*.wav")):
        if os.path.getmtime(filepath) < cutoff:
            os.remove(filepath)
            count += 1

    logger.info("Nettoyage audio: %d fichiers supprimés", count)
    return {"deleted": count}


# ─── Helpers privés ──────────────────────────────────────────────────────────

def _convert_to_wav(input_path: str) -> str:
    """Convertit n'importe quel format audio en WAV 16kHz mono (requis par Whisper)."""
    import subprocess

    if input_path.endswith(".wav"):
        return input_path  # Déjà en WAV

    output_path = input_path.rsplit(".", 1)[0] + "_16k.wav"
    cmd = [
        "ffmpeg", "-i", input_path,
        "-ar", "16000",  # Sample rate 16kHz
        "-ac", "1",      # Mono
        "-c:a", "pcm_s16le",  # Format PCM 16-bit
        output_path,
        "-y",  # Écraser si existe
        "-loglevel", "error",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")
    return output_path


def _compute_fluency_score(transcription: str, duration_sec: float) -> float:
    """
    Calcule un score de fluidité basé sur :
    - Débit de parole (mots par minute)
    - Ratio de pauses (estimé via longueur vs durée)
    """
    if not transcription or duration_sec <= 0:
        return 50.0

    words = transcription.split()
    wpm = (len(words) / duration_sec) * 60

    # Débit idéal pour l'anglais : 120-180 wpm
    if 120 <= wpm <= 180:
        score = 90.0
    elif 90 <= wpm < 120 or 180 < wpm <= 210:
        score = 75.0
    elif 60 <= wpm < 90 or 210 < wpm <= 240:
        score = 60.0
    else:
        score = 40.0

    # Pénalité pour transcription très courte (< 5 mots)
    if len(words) < 5:
        score *= 0.7

    return round(score, 2)


def _get_session_history(session) -> list:
    """Récupère les 10 derniers échanges pour le contexte LLM."""
    return list(
        session.exchanges.filter(transcription__isnull=False)
        .exclude(transcription="")
        .order_by("-exchange_index")[:10]
        .values("ai_question", "transcription", "ai_response")
    )


def _cleanup_audio(filepath: str):
    """Supprime un fichier audio de façon sécurisée."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except OSError as e:
        logger.warning("Impossible de supprimer %s: %s", filepath, e)
