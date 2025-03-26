import kagglehub

# Download latest version
path = kagglehub.dataset_download("wcukierski/enron-email-dataset")

print("Path to dataset files:", path)

pip install gensim sklearn pandas nltk plotly


import numpy as np
import pandas as pd
import plotly.express as px
import nltk
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

nltk.download('punkt')


# Example corpus (replace with real email data)
corpus = [
    ["free", "win", "lottery", "prize", "click", "here"],
    ["meeting", "schedule", "update", "team", "project"],
    ["urgent", "claim", "your", "reward", "now"],
    ["hello", "how", "are", "you", "today"]
]

# Train Word2Vec Model
word2vec_model = Word2Vec(sentences=corpus, vector_size=50, window=5, min_count=1, workers=4)

# Get words and their embeddings
words = list(word2vec_model.wv.index_to_key)
word_vectors = np.array([word2vec_model.wv[word] for word in words])



# Reduce word vectors to 2D using PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(word_vectors)

# Convert to DataFrame for visualization
df_pca = pd.DataFrame({"Word": words, "X": pca_result[:, 0], "Y": pca_result[:, 1]})

# Plot using Plotly
fig = px.scatter(df_pca, x="X", y="Y", text="Word", title="Word Embeddings - PCA")
fig.update_traces(textposition="top center")
fig.show()


# Reduce to 2D using t-SNE
tsne = TSNE(n_components=2, perplexity=3, random_state=42)
tsne_result = tsne.fit_transform(word_vectors)

# Convert to DataFrame for visualization
df_tsne = pd.DataFrame({"Word": words, "X": tsne_result[:, 0], "Y": tsne_result[:, 1]})

# Plot using Plotly
fig = px.scatter(df_tsne, x="X", y="Y", text="Word", title="Word Embeddings - t-SNE")
fig.update_traces(textposition="top center")
fig.show()

