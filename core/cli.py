import os
from ir_engine import (
    search_tfidf_cosine,
    search_bm25,
    search_jaccard,
    search_embeddings,
    show_results,
    load_evaluation_files,
    evaluar_modelo,
)

MODELOS = {
    'tfidf':      (search_tfidf_cosine, 'TF-IDF + Coseno'),
    'bm25':       (search_bm25,         'BM25'),
    'jaccard':    (search_jaccard,       'Jaccard'),
    'embeddings': (search_embeddings,    'Embeddings (ChromaDB)'),
}

QUERIES_PATH = os.path.join(os.path.dirname(__file__), '..', 'evaluation', 'queries.txt')
QRELS_PATH   = os.path.join(os.path.dirname(__file__), '..', 'evaluation', 'qrels.txt')

def cmd_evaluar(k=10):
    """Evalua los 4 modelos con queries.txt + qrels.txt y muestra tabla comparativa."""
    print("\nCargando evaluation/queries.txt y evaluation/qrels.txt...")
    qrels = load_evaluation_files(QUERIES_PATH, QRELS_PATH)
    print(f"{len(qrels)} consultas de prueba cargadas\n")

    resultados = {}
    for nombre, (fn, label) in MODELOS.items():
        print(f"Evaluando {label}...")
        resultados[label] = evaluar_modelo(fn, qrels, k=k)

    # Tabla comparativa
    col_w = 22
    metricas = [f'Precision@{k}', f'Recall@{k}', 'MAP']
    header = f"{'Modelo':<{col_w}}" + "".join(f"{m:>14}" for m in metricas)
    print(f"\n{'='*len(header)}")
    print(header)
    print(f"{'='*len(header)}")
    for label, res in resultados.items():
        row = f"{label:<{col_w}}" + "".join(f"{res[m]:>14.4f}" for m in metricas)
        print(row)
    print(f"{'='*len(header)}\n")


print("=== Sistema de Recuperacion de Informacion ===")
print("Escribe tu consulta en texto libre. Comandos: 'evaluar' | 'salir'\n")

while True:
    query = input("Consulta: ").strip()

    if not query:
        continue

    if query.lower() == 'salir':
        print("Saliendo...")
        break

    if query.lower() == 'evaluar':
        cmd_evaluar()
        continue

    # Busqueda semantica en la base de datos vectorial (ChromaDB)
    show_results(search_embeddings(query), model_name='Embeddings (ChromaDB)')
    print()
