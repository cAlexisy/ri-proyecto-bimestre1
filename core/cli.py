"""
Interfaz de linea de comandos (CLI) — Sistema de Recuperacion de Informacion
=============================================================================
Permite al usuario realizar consultas en texto libre sobre el corpus Reuters.
Las busquedas usan recuperacion semantica con embeddings y ChromaDB.
El comando 'evaluar' compara los 4 modelos implementados usando metricas de IR.
"""

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

# Los 4 modelos disponibles para la evaluacion comparativa
MODELOS = {
    'tfidf':      (search_tfidf_cosine, 'TF-IDF + Coseno'),
    'bm25':       (search_bm25,         'BM25'),
    'jaccard':    (search_jaccard,       'Jaccard'),
    'embeddings': (search_embeddings,    'Embeddings (ChromaDB)'),
}

# Rutas a los archivos de evaluacion (relativas a la ubicacion de este script)
QUERIES_PATH = os.path.join(os.path.dirname(__file__), '..', 'evaluation', 'queries.txt')
QRELS_PATH   = os.path.join(os.path.dirname(__file__), '..', 'evaluation', 'qrels.txt')

def cmd_evaluar(k=10):
    """
    Carga las consultas de prueba y documentos relevantes desde los archivos
    de evaluacion, ejecuta los 4 modelos y muestra una tabla comparativa
    con Precision@k, Recall@k y MAP para cada uno.
    """
    print("\nCargando evaluation/queries.txt y evaluation/qrels.txt...")
    qrels = load_evaluation_files(QUERIES_PATH, QRELS_PATH)
    print(f"{len(qrels)} consultas de prueba cargadas\n")

    # Evaluar cada modelo y guardar sus resultados
    resultados = {}
    for nombre, (fn, label) in MODELOS.items():
        print(f"Evaluando {label}...")
        resultados[label] = evaluar_modelo(fn, qrels, k=k)

    # Imprimir tabla comparativa
    col_w    = 22
    metricas = [f'Precision@{k}', f'Recall@{k}', 'MAP']
    header   = f"{'Modelo':<{col_w}}" + "".join(f"{m:>14}" for m in metricas)
    sep      = "=" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for label, res in resultados.items():
        print(f"{label:<{col_w}}" + "".join(f"{res[m]:>14.4f}" for m in metricas))
    print(f"{sep}\n")


# =============================================================================
# LOOP PRINCIPAL
# =============================================================================
print("=== Sistema de Recuperacion de Informacion ===")
print("Escribe tu consulta en texto libre. Comandos: 'evaluar' | 'salir'\n")

while True:
    query = input("Consulta: ").strip()

    if not query:
        continue

    # Salir del programa
    if query.lower() == 'salir':
        print("Saliendo...")
        break

    # Evaluar y comparar los 4 modelos con las consultas de prueba
    if query.lower() == 'evaluar':
        cmd_evaluar()
        continue

    # Para cualquier consulta libre, se usa recuperacion semantica con embeddings.
    # Los embeddings entienden el significado de las palabras, no solo coincidencias exactas.
    show_results(search_embeddings(query), model_name='Embeddings (ChromaDB)')
    print()
