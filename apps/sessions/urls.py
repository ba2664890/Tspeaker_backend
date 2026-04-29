"""T.Speak — URLs Sessions"""
from django.urls import path
from . import views

urlpatterns = [
    path("start/", views.SessionStartView.as_view(), name="session-start"),
    path("<uuid:session_id>/audio/", views.AudioUploadView.as_view(), name="session-audio"),
    path("<uuid:pk>/", views.SessionDetailView.as_view(), name="session-detail"),
    path("<uuid:session_id>/end/", views.end_session, name="session-end"),
    path("history/", views.SessionHistoryView.as_view(), name="session-history"),
    path("exchanges/<uuid:exchange_id>/result/", views.AudioResultView.as_view(), name="exchange-result"),
]
