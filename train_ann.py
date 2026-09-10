from pathlib import Path
import kagglehub
import numpy as np
import pandas as pd
import joblib

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Download and load Kaggle dataset
print("Downloading 1.2M Spotify dataset...")
dataset_dir = kagglehub.dataset_download("rodolfofigueroa/spotify-12m-songs")
csv_path = Path(dataset_dir) / "tracks_features.csv"

print("Reading CSV...")
feature_cols = [
    "danceability", "energy", "key", "loudness", 
    "mode", "speechiness", "acousticness", 
    "instrumentalness", "liveness", "valence", "tempo"
]
use_cols = ["id", "name", "artists"] + feature_cols

df = pd.read_csv(csv_path, usecols=use_cols)

# Optional: Sample to 150k songs if you want fast training on limited CPU/GPU RAM
# df = df.sample(n=150000, random_state=42).reset_index(drop=True)

# 2. Clean metadata for Django compatibility
df = df.rename(columns={
    "id": "track_id",
    "name": "track_name",
    "artists": "track_artist"
})

df["track_artist"] = (
    df["track_artist"]
    .astype(str)
    .str.strip("[]'")
    .str.replace("', '", ", ", regex=False)
)

df = df.dropna(subset=["track_id", "track_name", "track_artist"] + feature_cols)
df = df.drop_duplicates(subset=["track_id"]).reset_index(drop=True)

# 3. Scale audio features
print("Scaling audio features...")
scaler = StandardScaler()
X = scaler.fit_transform(df[feature_cols].to_numpy(dtype="float32"))

X_train, X_val = train_test_split(X, test_size=0.1, random_state=42)

# 4. Build the ANN Autoencoder architecture
input_dim = len(feature_cols)
embedding_dim = 16  # Dimensionality of the learned song representations

# Encoder
input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(64, activation="relu")(input_layer)
encoded = layers.BatchNormalization()(encoded)
encoded = layers.Dense(32, activation="relu")(encoded)
latent = layers.Dense(embedding_dim, activation="linear", name="latent_embedding")(encoded)

# Decoder
decoded = layers.Dense(32, activation="relu")(latent)
decoded = layers.Dense(64, activation="relu")(decoded)
output_layer = layers.Dense(input_dim, activation="linear")(decoded)

# Compile models
autoencoder = models.Model(inputs=input_layer, outputs=output_layer)
encoder = models.Model(inputs=input_layer, outputs=latent)

autoencoder.compile(optimizer="adam", loss="mse")

# 5. Train the ANN
print("Training Neural Network...")
autoencoder.fit(
    X_train,
    X_train,
    epochs=15,
    batch_size=512,
    shuffle=True,
    validation_data=(X_val, X_val),
    verbose=1
)

# 6. Generate embeddings for the entire dataset
print("Generating song embeddings...")
embeddings = encoder.predict(X, batch_size=1024).astype("float32")

# 7. Save output bundle for backend/api.py
output_dir = Path("spotify_ann_output")
output_dir.mkdir(parents=True, exist_ok=True)

bundle = {
    "songs": df[["track_id", "track_name", "track_artist"]].reset_index(drop=True),
    "embeddings": embeddings,
}

print("Saving model bundle to disk...")
joblib.dump(bundle, output_dir / "recommendation_data.joblib")
print(f"Done! Saved {len(df)} songs and embeddings to recommendation_data.joblib.")