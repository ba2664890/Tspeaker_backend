"""T.Speak — Vues Scoring"""
from django.core.cache import cache
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.db.models import Avg
from .models import Score


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def session_scores(request, session_id):
    try:
        score = Score.objects.get(session_id=session_id, user=request.user)
    except Score.DoesNotExist:
        return Response({"success": False, "error": {"message": "Score introuvable."}}, status=404)
    return Response({"success": True, "data": {
        "pronunciation": float(score.pronunciation),
        "fluency": float(score.fluency),
        "grammar": float(score.grammar),
        "vocabulary": float(score.vocabulary),
        "global_score": float(score.global_score),
        "feedback_text": score.feedback_text,
    }})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def user_stats(request):
    user = request.user
    return Response({"success": True, "data": {
        "avg_pronunciation": float(user.avg_pronunciation),
        "avg_fluency": float(user.avg_fluency),
        "avg_grammar": float(user.avg_grammar),
        "avg_vocabulary": float(user.avg_vocabulary),
        "sessions_count": user.sessions_count,
        "streak_days": user.streak_days,
        "xp_total": user.xp_total,
        "level": user.level,
    }})
