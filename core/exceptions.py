from rest_framework.views import exception_handler
from rest_framework.response import Response
from rest_framework import status
import logging

logger = logging.getLogger("tspeak.api")

def custom_exception_handler(exc, context):
    """
    Gestionnaire d'exceptions personnalisé pour DRF.
    Assure une réponse JSON cohérente même en cas d'erreur fatale.
    """
    # Appeler le gestionnaire par défaut de DRF pour obtenir la réponse standard
    response = exception_handler(exc, context)

    # Si DRF n'a pas pu gérer l'exception (ex: Erreur 500 inattendue)
    if response is None:
        logger.error(f"Exception non gérée: {str(exc)}", exc_info=True)
        return Response(
            {
                "error": "Une erreur serveur interne est survenue.",
                "detail": str(exc) if True else None, # On pourrait masquer en prod
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    # Enrichir la réponse d'erreur de DRF si nécessaire
    if response.status_code >= 400:
        logger.warning(f"Erreur API ({response.status_code}): {response.data}")
    
    return response
