import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Cargar el corpus
df = pd.read_csv("ModApte_test.csv")

# Eliminar filas con texto nulo
df = df.dropna(subset=['text'])
print(f"Documentos después de limpiar nulos: {len(df)}")

# Preprocesamiento
stop_words = set(stopwords.words('english'))

def preprocess(text):
    # Minúsculas
    text = text.lower()
    # Tokenizar
    tokens = word_tokenize(text)
    # Eliminar puntuación, stopwords y tokens cortos
    tokens = [
        t.replace('.', '').replace('-', ' ').strip()
        for t in tokens
        if t not in string.punctuation
        and t not in stop_words
        and len(t) > 2
    ]
    # Filtrar tokens vacíos que quedaron después del replace
    tokens = [t for t in tokens if t]
    return tokens

# Probar con el primer documento
print("\nTexto original:")
print(df['text'].iloc[0][:200])
print("\nTokens procesados:")
print(preprocess(df['text'].iloc[0])[:20])