# Sistema de Recuperación de Información
### Proyecto Bimestre 1 — Recuperación de Información

## Integrantes
- Alexis Chacon
- Kevin Alvear
- Miguel Muzo

---

## Descripción
Sistema que permite comparar cuatro modelos de recuperación de información
sobre el corpus Reuters-21578 (ModApte split):

- **Jaccard** — similitud binaria entre conjuntos de términos
- **TF-IDF + Coseno** — pesos por frecuencia inversa de documento
- **BM25** — modelo probabilístico con parámetros k1 y b
- **Embeddings semánticos** — sentence-transformers + ChromaDB

---

## Estructura del proyecto
    Proyecto-B1/
    ├── core/
    │   ├── tfidf_bm25.ipynb
    │   ├── ir_engine.py
    │   └── cli.py
    ├── data/
    │   └── ModApte_test.csv
    ├── evaluation/
    ├── requirements.txt
    └── README.md

---

## Requisitos
- Python 3.11+
- pip

---

## Instalación

**1. Clonar el repositorio**

    git clone https://github.com/cAlexisy/ri-proyecto-bimestre1.git
    cd ri-proyecto-bimestre1

**2. Crear entorno virtual**

    python -m venv .venv

**3. Activar el entorno virtual**

Windows:

    .venv\Scripts\Activate.ps1

Linux/Mac:

    source .venv/bin/activate

**4. Instalar dependencias**

    pip install -r requirements.txt

---

## Ejecución

### Interfaz de línea de comandos (CLI)

    cd core
    python cli.py

Escribe tu consulta en texto libre. Las búsquedas usan recuperación semántica (ChromaDB).

Comandos especiales:
- `evaluar` — evalúa los 4 modelos con qrels y muestra Precision@k, Recall@k y MAP
- `salir` — termina el programa

### Notebook completo
Abrir core/tfidf_bm25.ipynb en Jupyter o VS Code y ejecutar todas las celdas en orden.

---

## Corpus
Reuters-21578 — colección de artículos de noticias financieras.
Split utilizado: ModApte test (3,023 documentos tras limpieza).

---

## Modelos implementados

| Modelo | Descripción |
|--------|-------------|
| Jaccard | Similitud entre conjuntos de términos (vectores binarios) |
| TF-IDF + Coseno | Ponderación por frecuencia e inversión de documentos |
| BM25 | Modelo probabilístico con parámetros k1=1.5, b=0.75 |
| Embeddings | all-MiniLM-L6-v2 + búsqueda vectorial con ChromaDB |

---

## Evaluación
El sistema calcula para cada modelo:
- Precision@k y Recall@k por consulta
- MAP (Mean Average Precision) global
