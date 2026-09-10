from functools import lru_cache, wraps
from pathlib import Path

import joblib
import numpy as np

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.http import require_GET
from sklearn.metrics.pairwise import cosine_similarity


MODEL_PATH = (
    Path(settings.BASE_DIR).parent
    / "spotify_ann_output"
    / "recommendation_data.joblib"
)


@lru_cache(maxsize=1)
def load_model_data():
    bundle = joblib.load(MODEL_PATH)

    songs = bundle["songs"].reset_index(drop=True)
    embeddings = np.asarray(
        bundle["embeddings"],
        dtype="float32",
    )

    required = {"track_id", "track_name", "track_artist"}

    if not required.issubset(songs.columns):
        raise ValueError("The song catalog is missing required columns.")

    if (
        embeddings.ndim != 2
        or len(embeddings) != len(songs)
        or not np.isfinite(embeddings).all()
    ):
        raise ValueError("The saved embeddings are invalid.")

    return songs, embeddings


def api_endpoint(function):
    @wraps(function)
    @require_GET
    def wrapper(request):
        try:
            songs, embeddings = load_model_data()
        except Exception:
            return JsonResponse(
                {
                    "ready": False,
                    "error": (
                        "Could not load the model. Run spotify_ann.py "
                        "first and check recommendation_data.joblib."
                    ),
                },
                status=503,
            )

        try:
            result = function(request, songs, embeddings)

            return JsonResponse(
                result,
                json_dumps_params={
                    "ensure_ascii": False,
                    "allow_nan": False,
                },
            )

        except ValueError as error:
            return JsonResponse(
                {"error": str(error)},
                status=400,
            )

        except LookupError as error:
            return JsonResponse(
                {"error": str(error)},
                status=404,
            )

    return wrapper


def song_details(row):
    track_id = str(row["track_id"])

    return {
        "track_id": track_id,
        "track_name": str(row["track_name"]),
        "track_artist": str(row["track_artist"]),
        "spotify_url": (
            "https://open.spotify.com/track/" + track_id
        ),
    }


def get_limit(request, default, maximum):
    try:
        limit = int(request.GET.get("limit", default))
    except ValueError:
        raise ValueError("limit must be an integer.")

    if not 1 <= limit <= maximum:
        raise ValueError(
            f"limit must be between 1 and {maximum}."
        )

    return limit


@api_endpoint
def health(request, songs, embeddings):
    return {
        "ready": True,
        "song_count": len(songs),
    }


@api_endpoint
def search_songs(request, songs, embeddings):
    query = request.GET.get("q", "").strip()
    limit = get_limit(request, default=10, maximum=50)

    if len(query) > 200:
        raise ValueError("Search must be 200 characters or fewer.")

    names = songs["track_name"].astype(str).str.casefold()
    artists = songs["track_artist"].astype(str).str.casefold()

    matches = songs.loc[
        songs["track_id"].eq(query)
        | names.str.contains(query.casefold(), regex=False)
        | artists.str.contains(query.casefold(), regex=False)
    ]

    return {
        "total": len(matches),
        "songs": [
            song_details(row)
            for _, row in matches.head(limit).iterrows()
        ],
    }


@api_endpoint
def recommend_songs(request, songs, embeddings):
    track_id = request.GET.get("track_id", "").strip()
    limit = get_limit(request, default=5, maximum=20)

    if not track_id:
        raise ValueError("track_id is required.")

    positions = np.flatnonzero(
        songs["track_id"].eq(track_id).to_numpy()
    )

    if len(positions) == 0:
        raise LookupError("Song not found in the catalog.")

    index = int(positions[0])
    current_song = songs.iloc[index]

    similarities = cosine_similarity(
        embeddings[index:index + 1],
        embeddings,
    )[0]

    names = songs["track_name"].astype(str).str.casefold()
    artists = songs["track_artist"].astype(str).str.casefold()

    same_recording = (
        names.eq(str(current_song["track_name"]).casefold())
        & artists.eq(
            str(current_song["track_artist"]).casefold()
        )
    )

    similarities[same_recording.to_numpy()] = -np.inf

    recommendations = []
    seen = set()

    for position in np.argsort(-similarities, kind="stable"):
        if not np.isfinite(similarities[position]):
            continue

        identity = (
            names.iloc[position],
            artists.iloc[position],
        )

        if identity in seen:
            continue

        seen.add(identity)

        song = song_details(songs.iloc[position])
        song["similarity"] = round(
            float(np.clip(similarities[position], -1, 1)),
            4,
        )

        recommendations.append(song)

        if len(recommendations) == limit:
            break

    return {
        "current_song": song_details(current_song),
        "recommendations": recommendations,
    }