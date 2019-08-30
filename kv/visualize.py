import gensim
from gensim.models import Word2Vec, KeyedVectors
import pandas as pd
from sklearn.manifold import TSNE
import re
import matplotlib.pyplot as plt


model = KeyedVectors.load('paper6k-64d.kv')

vocab = list(model.vocab)
X = model[vocab]

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])

fig, axs = plt.subplots(1,1,figsize=(100,100))
axs.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    axs.annotate(word, pos)

fig.savefig('full_figure.png')
