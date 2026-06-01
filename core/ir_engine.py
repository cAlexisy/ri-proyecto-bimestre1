import re
import math
import string
import pandas as pd
import nltk
import chromadb
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer

nltk.download('punkt',     quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ── Corpus ────────────────────────────────────────────────────────────────────
df = pd.read_csv("../data/ModApte_test.csv")
df = df.dropna(subset=['text'])
df = df.reset_index(drop=True)

# ── Preprocesamiento ──────────────────────────────────────────────────────────
stop_words = set(stopwords.words('english'))

def preprocess(text):
    """Minusculas -> tokenizar -> quitar puntuacion/stopwords/tokens cortos."""
    tokens = word_tokenize(text.lower())
    tokens = [
        t.replace('.', '').replace('-', ' ').strip()
        for t in tokens
        if t not in string.punctuation and t not in stop_words and len(t) > 2
    ]
    return [t for t in tokens if t]

df['tokens']    = df['text'].apply(preprocess)
df['processed'] = df['tokens'].apply(lambda t: ' '.join(t))

# ── Indice invertido ──────────────────────────────────────────────────────────
# term -> {doc_id: frecuencia}
inverted_index = defaultdict(dict)
doc_len        = {}

for i, tokens in enumerate(df['tokens']):
    counts     = Counter(tokens)
    doc_len[i] = sum(counts.values())
    for term, freq in counts.items():
        inverted_index[term][i] = freq

doc_freq = {term: len(postings) for term, postings in inverted_index.items()}
N        = len(df)
avgdl    = sum(doc_len.values()) / N

# ── TF-IDF + coseno ───────────────────────────────────────────────────────────
vectorizer   = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['processed'])

def search_tfidf_cosine(query, top_k=10):
    """TF-IDF + similitud coseno."""
    q_vec  = vectorizer.transform([' '.join(preprocess(query))])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(doc_id, s) for doc_id, s in ranked[:top_k] if s > 0]

# ── BM25 (usa indice invertido) ───────────────────────────────────────────────
def search_bm25(query, top_k=10, k1=1.5, b=0.75):
    """BM25 con indice invertido — no itera sobre todos los documentos."""
    freq_q = Counter(preprocess(query))
    scores = defaultdict(float)

    for term in freq_q:
        postings = inverted_index.get(term)
        if not postings:
            continue
        df_t = doc_freq[term]
        idf  = math.log(1.0 + (N - df_t + 0.5) / (df_t + 0.5))
        for doc_id, tf in postings.items():
            dl    = doc_len[doc_id]
            denom = tf + k1 * (1.0 - b + b * dl / avgdl)
            scores[doc_id] += idf * (tf * (k1 + 1.0)) / denom

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

# ── Jaccard (candidatos via indice invertido) ─────────────────────────────────
def search_jaccard(query, top_k=10):
    """Jaccard — candidatos del indice invertido, sin fuerza bruta."""
    query_set = set(preprocess(query))
    if not query_set:
        return []

    candidates = set()
    for term in query_set:
        candidates.update(inverted_index.get(term, {}).keys())

    scores = []
    for doc_id in candidates:
        doc_set = set(df['tokens'].iloc[doc_id])
        union   = query_set | doc_set
        sim     = len(query_set & doc_set) / len(union) if union else 0.0
        if sim > 0:
            scores.append((doc_id, sim))

    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

# ── Embeddings + ChromaDB ─────────────────────────────────────────────────────
embedder      = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()
collection    = chroma_client.get_or_create_collection("corpus")

if collection.count() == 0:
    print("Indexando embeddings (puede tardar ~2 min)...")
    BATCH = 100
    for i in range(0, len(df), BATCH):
        chunk      = df['processed'].iloc[i:i+BATCH].tolist()
        embeddings = embedder.encode(chunk, normalize_embeddings=True).tolist()
        collection.add(
            documents  = chunk,
            embeddings = embeddings,
            ids        = [str(j) for j in range(i, i + len(chunk))]
        )

def search_embeddings(query, top_k=10):
    """Busqueda semantica con sentence-transformers + ChromaDB."""
    q_emb   = embedder.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=top_k)
    doc_ids = [int(i) for i in results['ids'][0]]
    scores  = [1.0 - d for d in results['distances'][0]]
    return list(zip(doc_ids, scores))

# ── Mostrar resultados ────────────────────────────────────────────────────────
def show_results(ranked_docs, model_name='Modelo', top_n=5):
    """Imprime ranking con score, titulo y snippet."""
    print(f'\n[{model_name}] Top {min(top_n, len(ranked_docs))} resultados')
    if not ranked_docs:
        print('Sin resultados.')
        return
    for rank, (doc_id, score) in enumerate(ranked_docs[:top_n], 1):
        row     = df.iloc[doc_id]
        title   = str(row['title']) if pd.notna(row['title']) else 'Sin titulo'
        text    = str(row['text']).replace('\n', ' ')
        snippet = text[:140] + ('...' if len(text) > 140 else '')
        print(f"  {rank}. [{score:.4f}] {title}")
        print(f"     {snippet}\n")

# ── Evaluacion ────────────────────────────────────────────────────────────────
def load_evaluation_files(queries_path, qrels_path):
    """
    Lee queries.txt y qrels.txt en formato TREC.
      queries.txt : '<QID> <texto>'       ej: Q1 oil prices
      qrels.txt   : '<QID> <doc_id> <1>'  ej: Q1 1331 1
    Retorna dict: {texto_consulta -> set(doc_ids relevantes)}
    compatible con evaluar_modelo().
    """
    queries = {}
    with open(queries_path, encoding='utf-8-sig') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                queries[parts[0]] = parts[1]

    qrels_by_qid = defaultdict(set)
    with open(qrels_path, encoding='utf-8-sig') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                qrels_by_qid[parts[0]].add(int(parts[1]))

    # Unir: texto_consulta -> set(doc_ids)
    return {queries[qid]: docs for qid, docs in qrels_by_qid.items() if qid in queries}

def build_qrels(min_docs=2):
    """
    Construye qrels desde la columna 'topics' del corpus.
    Retorna dict: topic -> set de doc_ids relevantes.
    """
    qrels = defaultdict(set)
    for idx in range(len(df)):
        valor = df.iloc[idx]['topics']
        if pd.isna(valor):
            continue
        for topic in re.findall(r"'([^']+)'", str(valor)):
            topic = topic.strip()
            if topic:
                qrels[topic].add(idx)
    return {t: docs for t, docs in qrels.items() if len(docs) >= min_docs}

def precision_at_k(ranked, relevant, k):
    top  = [d for d, _ in ranked[:k]]
    hits = sum(1 for d in top if d in relevant)
    return hits / k if k > 0 else 0.0

def recall_at_k(ranked, relevant, k):
    top  = [d for d, _ in ranked[:k]]
    hits = sum(1 for d in top if d in relevant)
    return hits / len(relevant) if relevant else 0.0

def average_precision(ranked, relevant):
    if not relevant:
        return 0.0
    hits, ap = 0, 0.0
    for rank, (doc_id, _) in enumerate(ranked, 1):
        if doc_id in relevant:
            hits += 1
            ap   += hits / rank
    return ap / len(relevant)

def evaluar_modelo(search_fn, qrels, k=10):
    """Evalua un modelo sobre todos los topics. Retorna Precision@k, Recall@k, MAP."""
    p_list, r_list, ap_list = [], [], []
    for topic, relevant in qrels.items():
        ranked = search_fn(topic, top_k=k)
        p_list.append(precision_at_k(ranked, relevant, k))
        r_list.append(recall_at_k(ranked, relevant, k))
        ap_list.append(average_precision(ranked, relevant))
    return {
        f'Precision@{k}': round(sum(p_list)  / len(p_list),  4),
        f'Recall@{k}':    round(sum(r_list)  / len(r_list),  4),
        'MAP':            round(sum(ap_list) / len(ap_list), 4),
    }
