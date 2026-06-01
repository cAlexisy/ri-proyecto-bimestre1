# Importar todas las funciones del motor de recuperación
from ir_engine import (
    search_tfidf_cosine,
    search_bm25,
    search_jaccard,
    search_embeddings,
    show_results
)

# ── Interfaz de línea de comandos ──────────────────────────────────────────────
print("=== Sistema de Recuperacion de Informacion ===")
print("Modelos disponibles: tfidf | bm25 | jaccard | embeddings")
print("Escribe 'salir' para terminar\n")

while True:
    # Leer la consulta del usuario
    query = input("Consulta: ").strip()

    # Salir si el usuario escribe 'salir'
    if query.lower() == 'salir':
        print("Saliendo...")
        break

    # Ignorar consultas vacías
    if not query:
        continue

    # Leer el modelo a usar
    modelo = input("Modelo [tfidf/bm25/jaccard/embeddings]: ").strip().lower()

    # Ejecutar el modelo seleccionado y mostrar resultados
    if modelo == 'tfidf':
        show_results(search_tfidf_cosine(query), model_name='TF-IDF + Coseno')
    elif modelo == 'bm25':
        show_results(search_bm25(query), model_name='BM25')
    elif modelo == 'jaccard':
        show_results(search_jaccard(query), model_name='Jaccard')
    elif modelo == 'embeddings':
        show_results(search_embeddings(query), model_name='Embeddings (ChromaDB)')
    else:
        print(f"Modelo '{modelo}' no reconocido. Usa: tfidf | bm25 | jaccard | embeddings")

    print()