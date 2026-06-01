import pandas as pd
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ── Carga del corpus ──────────────────────────────────────
df = pd.read_csv("data/ModApte_test.csv")
df = df.dropna(subset=['text'])
df = df.reset_index(drop=True)

# ── Preprocesamiento ──────────────────────────────────────
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [
        t.replace('.', '').replace('-', ' ').strip()
        for t in tokens
        if t not in string.punctuation
        and t not in stop_words
        and len(t) > 2
    ]
    return [t for t in tokens if t]

df['tokens'] = df['text'].apply(preprocess)
df['processed'] = df['tokens'].apply(lambda t: ' '.join(t))

# ── TF-IDF ────────────────────────────────────────────────
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['processed'])

def search_tfidf_cosine(query, top_k=10):
    query_vec = vectorizer.transform([' '.join(preprocess(query))])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(doc_id, score) for doc_id, score in ranked[:top_k] if score > 0]

# ── BM25 ──────────────────────────────────────────────────
from collections import defaultdict
import math

N = len(df)
avgdl = df['tokens'].apply(len).mean()
df_term = defaultdict(int)
for tokens in df['tokens']:
    for t in set(tokens):
        df_term[t] += 1

def bm25_score(query_tokens, doc_tokens, k1=1.5, b=0.75):
    score = 0
    dl = len(doc_tokens)
    freq = defaultdict(int)
    for t in doc_tokens:
        freq[t] += 1
    for t in query_tokens:
        if t not in freq:
            continue
        idf = math.log((N - df_term[t] + 0.5) / (df_term[t] + 0.5) + 1)
        tf = (freq[t] * (k1 + 1)) / (freq[t] + k1 * (1 - b + b * dl / avgdl))
        score += idf * tf
    return score

def search_bm25(query, top_k=10):
    query_tokens = preprocess(query)
    scores = [(i, bm25_score(query_tokens, tokens)) for i, tokens in enumerate(df['tokens'])]
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    return [(doc_id, score) for doc_id, score in ranked[:top_k] if score > 0]

# ── Jaccard ───────────────────────────────────────────────
def search_jaccard(query, top_k=10):
    query_set = set(preprocess(query))
    scores = []
    for i, tokens in enumerate(df['tokens']):
        doc_set = set(tokens)
        union = query_set | doc_set
        if union:
            scores.append((i, len(query_set & doc_set) / len(union)))
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

# ── Embeddings ────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
import chromadb

embedder = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("corpus")

if collection.count() == 0:
    print("Indexando embeddings...")
    batch = 100
    for i in range(0, len(df), batch):
        chunk = df['processed'].iloc[i:i+batch].tolist()
        embeddings = embedder.encode(chunk, normalize_embeddings=True).tolist()
        collection.add(
            documents=chunk,
            embeddings=embeddings,
            ids=[str(j) for j in range(i, i+len(chunk))]
        )

def search_embeddings(query, top_k=10):
    q_emb = embedder.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=top_k)
    doc_ids = [int(i) for i in results['ids'][0]]
    scores = [1 - d for d in results['distances'][0]]
    return list(zip(doc_ids, scores))

# ── Mostrar resultados ────────────────────────────────────
def show_results(ranked_docs, model_name='Modelo', top_n=5):
    print(f'\n[{model_name}] Top {min(top_n, len(ranked_docs))} resultados')
    if not ranked_docs:
        print('No se encontraron resultados.')
        return
    for rank, (doc_id, score) in enumerate(ranked_docs[:top_n], start=1):
        row = df.iloc[doc_id]
        title = str(row['title']) if pd.notna(row['title']) else 'Sin título'
        text = str(row['text']).replace('\n', ' ')
        snippet = text[:140] + ('...' if len(text) > 140 else '')
        print(f"{rank}. [{score:.4f}] {title}")
        print(f"   {snippet}\n")