"""T.Speak — Middleware et gestion d'exceptions personnalisés"""

import time
import logging
from rest_framework.views import exception_handler
from rest_framework.response import Response
from rest_framework import status

logger = logging.getLogger("tspeak.auth")
perf_logger = logging.getLogger("tspeak.performance")


class RequestLoggingMiddleware:
    """Middleware de logging des requêtes avec métriques de performance."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        start_time = time.monotonic()
        response = self.get_response(request)
        duration_ms = (time.monotonic() - start_time) * 1000

        # Log des requêtes lentes (> 1 seconde)
        if duration_ms > 1000:
            perf_logger.warning(
                "Requête lente: %s %s — %.0fms (user=%s)",
                request.method,
                request.path,
                duration_ms,
                getattr(request.user, "email", "anonymous"),
            )

        # Log des tentatives d'auth échouées
        if response.status_code == 401 and request.path.startswith("/api/"):
            logger.warning(
                "Auth échouée: %s — IP=%s",
                request.path,
                request.META.get("HTTP_X_FORWARDED_FOR", request.META.get("REMOTE_ADDR")),
            )

        response["X-Response-Time"] = f"{duration_ms:.0f}ms"
        return response


def custom_exception_handler(exc, context):
    """Handler d'exceptions DRF avec format de réponse unifié T.Speak."""
    response = exception_handler(exc, context)

    if response is not None:
        error_data = {
            "success": False,
            "error": {
                "code": response.status_code,
                "message": _extract_message(response.data),
                "details": response.data if isinstance(response.data, dict) else {},
            },
        }
        response.data = error_data
    else:
        # Erreurs non gérées par DRF
        logger.error("Erreur non gérée: %s", exc, exc_info=True)
        response = Response(
            {
                "success": False,
                "error": {
                    "code": 500,
                    "message": "Une erreur interne s'est produite.",
                    "details": {},
                },
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    return response


def _extract_message(data):
    if isinstance(data, dict):
        if "detail" in data:
            return str(data["detail"])
        if "non_field_errors" in data:
            return str(data["non_field_errors"][0])
        return "Données invalides"
    if isinstance(data, list):
        return str(data[0]) if data else "Erreur de validation"
    return str(data)
