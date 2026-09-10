import { useEffect, useState } from "react";
import "./index.css";

async function requestApi(path, signal) {
  const response = await fetch(path, { signal });
  const data = await response.json().catch(() => null);

  if (!response.ok || !data) {
    throw new Error(
      data?.error || "Could not reach the API. Check that Django is running."
    );
  }
  return data;
}

export default function App() {
  const [query, setQuery] = useState("");
  const [songs, setSongs] = useState([]);
  const [currentSong, setCurrentSong] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    requestApi("/api/songs/?limit=10", controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) setSongs(data.songs);
      })
      .catch((error) => {
        if (!controller.signal.aborted) setError(error.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setBusy(false);
      });

    return () => controller.abort();
  }, []);

  async function searchSongs(event) {
    event.preventDefault();
    if (busy) return;

    setBusy(true);
    setError("");

    try {
      const params = new URLSearchParams({ q: query.trim(), limit: "10" });
      const data = await requestApi(`/api/songs/?${params}`);
      setSongs(data.songs);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function getRecommendations(song) {
    if (busy) return;

    setBusy(true);
    setError("");
    setCurrentSong(song);
    setRecommendations([]);

    try {
      const params = new URLSearchParams({ track_id: song.track_id, limit: "5" });
      const data = await requestApi(`/api/recommendations/?${params}`);
      setCurrentSong(data.current_song);
      setRecommendations(data.recommendations);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="app">
      <header>
        <p className="eyebrow">SPOTIFY ANN RECOMMENDER</p>
        <h1>Find your next song</h1>
        <p>Choose a song to discover tracks with similar audio features.</p>
      </header>

      <form className="search-form" onSubmit={searchSongs}>
        <label htmlFor="song-search" className="sr-only">
          Search by song title, artist, or track ID
        </label>
        <input
          id="song-search"
          type="search"
          placeholder="Search a song or artist..."
          value={query}
          maxLength={200}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button type="submit" disabled={busy}>Search</button>
      </form>

      {busy && <p role="status">Loading...</p>}
      {error && <p className="error" role="alert">{error}</p>}

      <div className="columns">
        <section className="panel">
          <h2>Choose your starting song</h2>
          {!busy && songs.length === 0 && !error && (
            <p>No songs found. Try another title or artist.</p>
          )}
          {songs.map((song) => (
            <button
              type="button"
              className="song-button"
              key={song.track_id}
              disabled={busy}
              aria-pressed={currentSong?.track_id === song.track_id}
              onClick={() => getRecommendations(song)}
            >
              <strong>{song.track_name}</strong>
              <span>{song.track_artist}</span>
            </button>
          ))}
        </section>

        <section className="panel">
          <h2>Recommended next songs</h2>
          {!currentSong && <p>Select a song from the list to begin.</p>}
          {currentSong && (
            <div className="current-song">
              <span>CURRENT SONG</span>
              <h3>{currentSong.track_name}</h3>
              <p>{currentSong.track_artist}</p>
            </div>
          )}
          {recommendations.map((song, index) => (
            <article className="recommendation" key={song.track_id}>
              <h3>{index + 1}. {song.track_name}</h3>
              <p>{song.track_artist}</p>
              <p className="score">Similarity: {song.similarity.toFixed(3)}</p>
              <div className="song-actions">
                <a href={song.spotify_url} target="_blank" rel="noopener noreferrer">
                  Open in Spotify
                </a>
                <button type="button" disabled={busy} onClick={() => getRecommendations(song)}>
                  Find similar songs
                </button>
              </div>
            </article>
          ))}
          {currentSong && !busy && !error && recommendations.length === 0 && (
            <p>No other distinct songs are available.</p>
          )}
        </section>
      </div>
    </main>
  );
}