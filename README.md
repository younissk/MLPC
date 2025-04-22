# MLPC2025 Dataset
- `metadata.csv` lists the individual audio files in the data set and corresponding metadata (keywords, descriptions, title, license, download link of the original file, ...)
- `metadata_keywords_embeddings.npz` holds one text embedding vector for each list of keywords in `metadata.csv`; rows of `metadata.csv` and `metadata_keywords_embeddings.npz` are aligned; use the index to retrieve the text embedding
- `metadata_title_embeddings.npz` holds one text embedding vector for each title in `metadata.csv`; rows of `metadata.csv` and `metadata_title_embeddings.npz` are aligned; use the index to retrieve the text embedding
- `annotation.csv` list all temporal annotations and the text description of the region
- `annotations_text_embeddings.npz` holds one text embedding vector for each annotation in `annotations.csv`; rows of `annotations.csv` and `annotations_text_embeddings.npz` are aligned; use the index to retrieve the text embedding
- folder `audio` contains the audio recordings in mp3 format
- folder `audio_features` contains the audio features we extracted for you from the waveforms; each feature file holds multiple feature array. 
  - See the example below to on how to access the individual arrays.

```python
import numpy as np
import pandas as pd
import os

# load the metadata
metadata_df = pd.read_csv("metadata.csv")
title_embeddings = np.load("metadata_title_embeddings.npz")["embeddings"]
keywords_embeddings = np.load("metadata_keywords_embeddings.npz")["embeddings"]

# load the annotations
annotations_df = pd.read_csv("annotations.csv")
annotations_embeddings = np.load("annotations_text_embeddings.npz")["embeddings"]

# load audio features
feature_filename = metadata_df.loc[0, "filename"].replace("mp3", "npz")
features = np.load(os.path.join("audio_features", feature_filename))
print(list(features.keys()))

print("Shape of ZCR feature (time, n_features)", features["zerocrossingrate"].shape)
print("Shape of MFCC features (time, n_features)", features["mfcc"].shape)

# load audio (optional, just i you want to compute your own features ...)
import librosa
waveform, sr = librosa.load(os.path.join("audio", metadata_df.loc[0, "filename"]), sr=16000)
```



# LICENSING
Find licenses and license holders of the individual audio files in `metadata.csv`.
The annotations, the metadata, and the precomputed features/ embeddings must not be used for purposes other than the MLCP course. 
**Do not share the dataset or the annotations online!**
