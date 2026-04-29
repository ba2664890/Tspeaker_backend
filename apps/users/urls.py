"""T.Speak — URLs Authentification"""

from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView
from . import views

urlpatterns = [
    path("register/", views.RegisterView.as_view(), name="auth-register"),
    path("login/", views.LoginView.as_view(), name="auth-login"),
    path("refresh/", TokenRefreshView.as_view(), name="auth-refresh"),
    path("logout/", views.LogoutView.as_view(), name="auth-logout"),
    path("me/", views.UserProfileView.as_view(), name="auth-me"),
    path("leaderboard/", views.LeaderboardView.as_view(), name="auth-leaderboard"),
    path("streak/", views.update_streak, name="auth-streak"),
]
