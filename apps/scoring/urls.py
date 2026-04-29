"""T.Speak — URLs Scoring"""
from django.urls import path
from . import views

urlpatterns = [
    path("<uuid:session_id>/", views.session_scores, name="session-scores"),
    path("stats/", views.user_stats, name="user-stats"),
]
