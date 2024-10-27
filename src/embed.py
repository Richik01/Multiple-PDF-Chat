from sentence_transformers import SentenceTransformer
import numpy as np

def embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings for the text chunks
    embeddings = model.encode(chunks, show_progress_bar=True)

    embeddings = np.array(embeddings)
    return embeddings