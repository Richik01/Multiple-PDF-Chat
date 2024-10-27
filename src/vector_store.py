import faiss
import pickle
def faiss_vs(embeddings, chunks, file_path):
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance metric
    index.add(embeddings)  # Add vectors to the index 
    with open(file_path, 'wb') as f:
        pickle.dump((index, chunks), f)
