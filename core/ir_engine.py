import pandas as pd
import nltk
import string
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Descarga silenciosa de recursos NLTK necesarios para tokenización y stopwords
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ── Carga del corpus ───────────────────────────────────────────────────────────
# Se lee el CSV del corpus Reuters-21578 desde la carpeta data/
df = pd.read_csv("../data/ModApte_test.csv")

# Eliminar filas que no tienen texto
df = df.dropna(subset=['text'])

# Resetear índices para que sean consecutivos desde 0
df = df.reset_index(drop=True)

# ── Preprocesamiento ───────────────────────────────────────────────────────────
# Cargar lista de stopwords en inglés
stop_words = set(stopwords.words('english'))

def preprocess(text):
    """
    Limpia y tokeniza un texto aplicando:
    - Conversión a minúsculas
    - Tokenización por palabras
    - Eliminación de puntuación y stopwords
    - Eliminación de tokens cortos (menos de 3 caracteres)
    """
    # Convertir a minúsculas
    text = text.lower()

    # Tokenizar el texto en palabras individuales
    tokens = word_tokenize(text)

    # Limpiar cada token: quitar puntos, guiones y filtrar stopwords y tokens cortos
    tokens = [
        t.replace('.', '').replace('-', ' ').strip()
        for t in tokens
        if t not in string.punctuation
        and t not in stop_words
        and len(t) > 2
    ]

    # Eliminar tokens que quedaron vacíos después del reemplazo
    return [t for t in tokens if t]

# Aplicar preprocesamiento a todos los documentos del corpus
df['tokens'] = df['text'].apply(preprocess)

# Unir tokens en un string para usar con TfidfVectorizer
df['processed'] = df['tokens'].apply(lambda t: ' '.join(t))

# ── Modelo TF-IDF ──────────────────────────────────────────────────────────────
# Crear y entrenar el vectorizador TF-IDF sobre el corpus completo
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['processed'])

def search_tfidf_cosine(query, top_k=10):
    """
    Recupera documentos usando TF-IDF + similitud coseno.
    - Vectoriza la consulta con el mismo vectorizador del corpus
    - Calcula similitud coseno entre la consulta y todos los documentos
    - Retorna los top_k documentos más similares
    """
    # Preprocesar y vectorizar la consulta
    query_vec = vectorizer.transform([' '.join(preprocess(query))])

    # Calcular similitud coseno con todos los documentos
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Ordenar por score descendente
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    # Retornar solo documentos con score mayor a 0
    return [(doc_id, score) for doc_id, score in ranked[:top_k] if score > 0]

# ── Modelo BM25 ────────────────────────────────────────────────────────────────
# Parámetros del modelo BM25
N = len(df)                                    # Total de documentos
avgdl = df['tokens'].apply(len).mean()         # Longitud promedio de documentos

# Calcular frecuencia de documentos por término (cuántos docs contienen cada término)
df_term = defaultdict(int)
for tokens in df['tokens']:
    for t in set(tokens):
        df_term[t] += 1

def bm25_score(query_tokens, doc_tokens, k1=1.5, b=0.75):
    """
    Calcula el score BM25 entre una consulta y un documento.
    k1: controla la saturación de la frecuencia del término
    b: controla la normalización por longitud del documento
    """
    score = 0
    dl = len(doc_tokens)  # Longitud del documento actual

    # Frecuencia de cada término en el documento
    freq = defaultdict(int)
    for t in doc_tokens:
        freq[t] += 1

    for t in query_tokens:
        if t not in freq:
            continue

        # IDF: penaliza términos muy comunes en el corpus
        idf = math.log((N - df_term[t] + 0.5) / (df_term[t] + 0.5) + 1)

        # TF normalizado por longitud del documento
        tf = (freq[t] * (k1 + 1)) / (freq[t] + k1 * (1 - b + b * dl / avgdl))

        score += idf * tf

    return score

def search_bm25(query, top_k=10):
    """
    Recupera documentos usando el modelo BM25.
    Retorna los top_k documentos ordenados por score descendente.
    """
    query_tokens = preprocess(query)

    # Calcular score BM25 para cada documento del corpus
    scores = [(i, bm25_score(query_tokens, tokens)) for i, tokens in enumerate(df['tokens'])]

    # Ordenar por score descendente
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)

    return [(doc_id, score) for doc_id, score in ranked[:top_k] if score > 0]

# ── Modelo Jaccard ─────────────────────────────────────────────────────────────
def search_jaccard(query, top_k=10):
    """
    Recupera documentos usando similitud Jaccard.
    Jaccard = |intersección| / |unión| entre el conjunto de términos
    de la consulta y el conjunto de términos del documento.
    """
    query_set = set(preprocess(query))
    scores = []

    for i, tokens in enumerate(df['tokens']):
        doc_set = set(tokens)
        union = query_set | doc_set

        if union:
            # Calcular similitud Jaccard
            similitud = len(query_set & doc_set) / len(union)
            scores.append((i, similitud))

    # Ordenar por similitud descendente y retornar top_k
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

# ── Modelo Embeddings ──────────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
import chromadb

# Cargar modelo preentrenado de sentence-transformers
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Inicializar cliente de ChromaDB en memoria
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("corpus")

# Indexar documentos si la colección está vacía
if collection.count() == 0:
    print("Indexando embeddings, esto puede tardar unos minutos...")
    batch = 100  # Procesar en lotes para evitar problemas de memoria

    for i in range(0, len(df), batch):
        chunk = df['processed'].iloc[i:i+batch].tolist()

        # Generar embeddings normalizados para el lote
        embeddings = embedder.encode(chunk, normalize_embeddings=True).tolist()

        # Agregar a ChromaDB con IDs únicos
        collection.add(
            documents=chunk,
            embeddings=embeddings,
            ids=[str(j) for j in range(i, i+len(chunk))]
        )

def search_embeddings(query, top_k=10):
    """
    Recupera documentos usando búsqueda semántica con embeddings.
    - Genera el embedding de la consulta
    - Busca los documentos más cercanos en ChromaDB
    - Convierte distancia coseno a similitud (1 - distancia)
    """
    # Generar embedding de la consulta
    q_emb = embedder.encode([query], normalize_embeddings=True).tolist()

    # Buscar los top_k documentos más similares en ChromaDB
    results = collection.query(query_embeddings=q_emb, n_results=top_k)

    # Convertir IDs y distancias a similitudes
    doc_ids = [int(i) for i in results['ids'][0]]
    scores = [1 - d for d in results['distances'][0]]

    return list(zip(doc_ids, scores))

# ── Mostrar resultados ─────────────────────────────────────────────────────────
def show_results(ranked_docs, model_name='Modelo', top_n=5):
    """
    Imprime los resultados de una búsqueda de forma legible.
    Muestra el ranking, score, título y un snippet del texto.
    """
    print(f'\n[{model_name}] Top {min(top_n, len(ranked_docs))} resultados')

    if not ranked_docs:
        print('No se encontraron resultados.')
        return

    for rank, (doc_id, score) in enumerate(ranked_docs[:top_n], start=1):
        row = df.iloc[doc_id]

        # Obtener título o marcar como sin título
        title = str(row['title']) if pd.notna(row['title']) else 'Sin título'

        # Limpiar saltos de línea del texto y recortar a 140 caracteres
        text = str(row['text']).replace('\n', ' ')
        snippet = text[:140] + ('...' if len(text) > 140 else '')

        print(f"{rank}. [{score:.4f}] {title}")
        print(f"   {snippet}\n")