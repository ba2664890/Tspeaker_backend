"""T.Speak — Validation des fichiers audio"""
from django.conf import settings


def validate_audio_file(audio_file) -> str | None:
    """
    Valide un fichier audio uploadé.
    Retourne un message d'erreur ou None si valide.
    """
    max_size = getattr(settings, "AUDIO_MAX_SIZE_MB", 10) * 1024 * 1024
    allowed_formats = getattr(settings, "AUDIO_ALLOWED_FORMATS", ["wav", "m4a", "mp3"])

    if audio_file.size > max_size:
        return f"Fichier trop grand ({audio_file.size // 1024 // 1024}MB). Maximum: {max_size // 1024 // 1024}MB"

    ext = audio_file.name.rsplit(".", 1)[-1].lower() if "." in audio_file.name else ""
    if ext not in allowed_formats:
        return f"Format non supporté: '{ext}'. Formats acceptés: {', '.join(allowed_formats)}"

    return None
