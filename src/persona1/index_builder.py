from indexer import df, preprocess

def build_index(df):
    inverted_index = {}
    doc_lengths = {}

    for i in range(len(df)):
        row = df.iloc[i]
        doc_id = row["doc_id"]
        tokens = preprocess(row["text"])
        doc_lengths[doc_id] = len(tokens)

        for token in tokens:
            if token not in inverted_index:
                inverted_index[token] = {}
            if doc_id not in inverted_index[token]:
                inverted_index[token][doc_id] = 0
            inverted_index[token][doc_id] += 1

    N = len(doc_lengths)
    return inverted_index, doc_lengths, N

if __name__ == "__main__":
    inverted_index, doc_lengths, N = build_index(df)
    print(f"Documentos indexados: {N}")
    print(f"Términos únicos: {len(inverted_index)}")

    term = list(inverted_index.keys())[0]
    print(f"\nEjemplo — término '{term}':")
    print(dict(list(inverted_index[term].items())[:3]))
