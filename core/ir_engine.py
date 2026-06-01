"""
Motor de Recuperacion de Informacion — Reuters-21578
====================================================
Este modulo contiene toda la logica del sistema:
  - Carga y limpieza del corpus
  - Indice invertido
  - Modelos de busqueda: Jaccard, TF-IDF, BM25, Embeddings
  - Funciones de evaluacion: Precision, Recall, MAP
"""

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

# Descarga silenciosa de recursos de NLTK necesarios para tokenizar
nltk.download('punkt',     quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# =============================================================================
# 1. CARGA DEL CORPUS
# =============================================================================
# Se carga el dataset Reuters-21578 (split ModApte, particion de prueba).
# Reuters es una coleccion clasica de noticias financieras usada en IR.
df = pd.read_csv("../data/ModApte_test.csv")

# Eliminar documentos que no tienen texto
df = df.dropna(subset=['text'])

# Reiniciar indices para que sean 0, 1, 2, ... sin huecos
df = df.reset_index(drop=True)

# =============================================================================
# 2. PREPROCESAMIENTO
# =============================================================================
# Las stopwords son palabras muy comunes que no aportan significado
# para buscar (ej: "the", "is", "and"). Las eliminamos para reducir ruido.
stop_words = set(stopwords.words('english'))

def preprocess(text):
    """
    Limpia un texto para dejarlo listo para indexar o buscar.
    Pasos: minusculas -> separar en palabras -> quitar puntuacion,
    stopwords y palabras muy cortas (menos de 3 letras).
    """
    # Todo en minusculas para que "Trade" y "trade" sean lo mismo
    tokens = word_tokenize(text.lower())

    # Limpiar cada palabra: quitar puntos, guiones y filtrar basura
    tokens = [
        t.replace('.', '').replace('-', ' ').strip()
        for t in tokens
        if t not in string.punctuation and t not in stop_words and len(t) > 2
    ]

    # Eliminar tokens que quedaron vacios tras la limpieza
    return [t for t in tokens if t]

# Aplicar preprocesamiento a cada documento del corpus
df['tokens']    = df['text'].apply(preprocess)

# Version en string (para TF-IDF que necesita texto, no lista)
df['processed'] = df['tokens'].apply(lambda t: ' '.join(t))

# =============================================================================
# 3. INDICE INVERTIDO
# =============================================================================
# El indice invertido es la estructura central de un motor de busqueda.
# Para cada palabra guarda en que documentos aparece y cuantas veces.
# Ejemplo: "oil" -> {doc5: 3, doc12: 1, doc47: 2}
# Asi, cuando alguien busca "oil", sabemos exactamente donde ir sin
# tener que leer todos los documentos uno por uno.

inverted_index = defaultdict(dict)  # term -> {doc_id: frecuencia}
doc_len        = {}                  # longitud de cada documento (en tokens)

for i, tokens in enumerate(df['tokens']):
    counts     = Counter(tokens)       # contar cuantas veces aparece cada palabra
    doc_len[i] = sum(counts.values())  # total de palabras en el documento
    for term, freq in counts.items():
        inverted_index[term][i] = freq

# Cuantos documentos contienen cada termino (para calcular IDF)
doc_freq = {term: len(postings) for term, postings in inverted_index.items()}

N     = len(df)                          # total de documentos en el corpus
avgdl = sum(doc_len.values()) / N        # longitud promedio de documento

# =============================================================================
# 4. MODELO TF-IDF + SIMILITUD COSENO
# =============================================================================
# TF-IDF pondera cada palabra segun que tan frecuente es en un documento
# (TF) pero penaliza las que aparecen en casi todos los documentos (IDF).
# La similitud coseno mide el angulo entre dos vectores: si apuntan en
# la misma direccion, los documentos son similares a la consulta.

vectorizer   = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['processed'])

def search_tfidf_cosine(query, top_k=10):
    """
    Busca documentos usando TF-IDF y similitud coseno.
    Convierte la consulta al mismo espacio vectorial del corpus
    y retorna los top_k documentos mas similares.
    """
    q_vec  = vectorizer.transform([' '.join(preprocess(query))])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(doc_id, s) for doc_id, s in ranked[:top_k] if s > 0]

# =============================================================================
# 5. MODELO BM25
# =============================================================================
# BM25 mejora TF-IDF con dos ajustes:
#   - Saturacion de frecuencia: si una palabra aparece 10 veces vs 5 veces,
#     no significa que el documento sea el doble de relevante.
#   - Normalizacion por longitud: un documento largo tiene mas palabras
#     por naturaleza, no necesariamente por ser mas relevante.
# k1=1.5 controla la saturacion, b=0.75 controla la normalizacion.

def search_bm25(query, top_k=10, k1=1.5, b=0.75):
    """
    Busca documentos usando el modelo BM25.
    Usa el indice invertido para consultar solo los documentos
    que contienen alguna palabra de la consulta (eficiente).
    """
    freq_q = Counter(preprocess(query))
    scores = defaultdict(float)

    for term in freq_q:
        # Solo procesar terminos que existen en el indice
        postings = inverted_index.get(term)
        if not postings:
            continue

        # IDF: que tan raro es este termino en el corpus
        df_t = doc_freq[term]
        idf  = math.log(1.0 + (N - df_t + 0.5) / (df_t + 0.5))

        # Para cada documento que contiene el termino, calcular su aporte
        for doc_id, tf in postings.items():
            dl    = doc_len[doc_id]
            denom = tf + k1 * (1.0 - b + b * dl / avgdl)
            scores[doc_id] += idf * (tf * (k1 + 1.0)) / denom

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

# =============================================================================
# 6. MODELO JACCARD
# =============================================================================
# Jaccard es el modelo mas simple: mide cuantas palabras comparten
# la consulta y el documento, dividido entre el total de palabras distintas.
# Ejemplo: consulta={oil, price}, doc={oil, gas, price, market}
#   interseccion=2, union=4, Jaccard=0.5
# No considera frecuencias ni importancia de palabras.

def search_jaccard(query, top_k=10):
    """
    Busca documentos usando similitud Jaccard.
    Primero obtiene candidatos del indice invertido (documentos que
    comparten al menos una palabra con la consulta) y luego calcula
    la similitud exacta solo para esos candidatos.
    """
    query_set = set(preprocess(query))
    if not query_set:
        return []

    # Candidatos: documentos que tienen al menos una palabra de la consulta
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

# =============================================================================
# 7. RECUPERACION SEMANTICA CON EMBEDDINGS + CHROMADB
# =============================================================================
# Los embeddings convierten texto en vectores numericos que capturan
# el significado, no solo las palabras exactas.
# Ejemplo: "car" y "automobile" tendran vectores muy cercanos aunque
# sean palabras distintas — algo que TF-IDF y BM25 no pueden hacer.
#
# Se usa el modelo all-MiniLM-L6-v2 (liviano y preciso) de
# sentence-transformers, y ChromaDB como base de datos vectorial
# para buscar los documentos mas cercanos de forma eficiente.

embedder      = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()
collection    = chroma_client.get_or_create_collection("corpus")

# Indexar el corpus en ChromaDB si aun no se ha hecho
if collection.count() == 0:
    print("Indexando embeddings (puede tardar ~2 min)...")
    BATCH = 100  # procesar de a 100 documentos para no saturar memoria
    for i in range(0, len(df), BATCH):
        chunk      = df['processed'].iloc[i:i+BATCH].tolist()
        embeddings = embedder.encode(chunk, normalize_embeddings=True).tolist()
        collection.add(
            documents  = chunk,
            embeddings = embeddings,
            ids        = [str(j) for j in range(i, i + len(chunk))]
        )

def search_embeddings(query, top_k=10):
    """
    Busqueda semantica: convierte la consulta en embedding y busca
    los documentos con vectores mas cercanos en ChromaDB.
    La distancia coseno se convierte a similitud: score = 1 - distancia.
    """
    q_emb   = embedder.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=top_k)
    doc_ids = [int(i) for i in results['ids'][0]]
    scores  = [1.0 - d for d in results['distances'][0]]
    return list(zip(doc_ids, scores))

# =============================================================================
# 8. MOSTRAR RESULTADOS
# =============================================================================
def show_results(ranked_docs, model_name='Modelo', top_n=5):
    """
    Imprime los resultados de una busqueda de forma legible:
    numero de ranking, score, titulo del articulo y un fragmento del texto.
    """
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

# =============================================================================
# 9. EVALUACION
# =============================================================================
# Para medir que tan bien funciona cada modelo se usan qrels:
# listas que dicen, para cada consulta de prueba, cuales documentos
# son realmente relevantes. Luego se mide cuantos de esos relevantes
# aparecen en los primeros k resultados del modelo.

def load_evaluation_files(queries_path, qrels_path):
    """
    Carga las consultas de prueba (queries.txt) y los documentos
    relevantes (qrels.txt) desde archivos en formato TREC estandar.
      queries.txt : 'Q1 oil prices'       (ID y texto de consulta)
      qrels.txt   : 'Q1 1331 1'           (ID, doc relevante, relevancia)
    Retorna un diccionario: {texto_consulta -> conjunto de doc_ids relevantes}
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

    return {queries[qid]: docs for qid, docs in qrels_by_qid.items() if qid in queries}

def build_qrels(min_docs=2):
    """
    Alternativa: construye qrels automaticamente desde los topics del corpus.
    Cada topic (ej: 'crude', 'trade') se usa como consulta, y los documentos
    etiquetados con ese topic son los relevantes.
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
    """
    Precision@k: de los primeros k resultados, que fraccion son relevantes.
    Ejemplo: si 3 de los 10 primeros son relevantes, Precision@10 = 0.3
    """
    top  = [d for d, _ in ranked[:k]]
    hits = sum(1 for d in top if d in relevant)
    return hits / k if k > 0 else 0.0

def recall_at_k(ranked, relevant, k):
    """
    Recall@k: de todos los documentos relevantes que existen,
    que fraccion aparece en los primeros k resultados.
    Ejemplo: si hay 20 relevantes y encontramos 5 en top-10, Recall@10 = 0.25
    """
    top  = [d for d, _ in ranked[:k]]
    hits = sum(1 for d in top if d in relevant)
    return hits / len(relevant) if relevant else 0.0

def average_precision(ranked, relevant):
    """
    Average Precision (AP): precision promedio calculada cada vez que
    aparece un documento relevante en el ranking.
    Premia a los modelos que ponen los relevantes mas arriba.
    """
    if not relevant:
        return 0.0
    hits, ap = 0, 0.0
    for rank, (doc_id, _) in enumerate(ranked, 1):
        if doc_id in relevant:
            hits += 1
            ap   += hits / rank
    return ap / len(relevant)

def evaluar_modelo(search_fn, qrels, k=10):
    """
    Evalua un modelo sobre todas las consultas de prueba.
    Para cada consulta ejecuta la busqueda y calcula las metricas.
    MAP (Mean Average Precision) es el promedio de AP sobre todas las consultas
    — es la metrica principal para comparar modelos de recuperacion.
    """
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
