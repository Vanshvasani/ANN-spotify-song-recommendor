from django.urls import path

from api import (
    health,
    search_songs,
    recommend_songs,
)


urlpatterns = [
    path("api/health/", health),
    path("api/songs/", search_songs),
    path("api/recommendations/", recommend_songs),
]