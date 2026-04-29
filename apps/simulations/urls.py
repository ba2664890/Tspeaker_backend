"""T.Speak — URLs Simulations"""
from django.urls import path
from .models import SimulationListView, SimulationDetailView, SimulationStartView

urlpatterns = [
    path("", SimulationListView.as_view(), name="simulation-list"),
    path("<uuid:pk>/", SimulationDetailView.as_view(), name="simulation-detail"),
    path("<uuid:pk>/start/", SimulationStartView.as_view(), name="simulation-start"),
]
