from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import num_tokens_from_string
import os
from openai import OpenAI
import numpy as np

class SemanticChunker():
    def __init__(self, max_chunk_size=400, min_chunk_size=50, BERT=False):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=min_chunk_size,
            chunk_overlap=0,
            length_function=num_tokens_from_string
            )
        
        self.max_cluster = max_chunk_size//min_chunk_size
        
    def _get_similarity_matrix(self, sentences):

        OPENAI_API_KEY = os.getenv('OPENAI_CHROMA_API_KEY')
        client = OpenAI(api_key=OPENAI_API_KEY)

        print(len(sentences))

        BATCH_SIZE = 500
        N = len(sentences)
        embedding_matrix = None

        for i in range(0, N, BATCH_SIZE):
            batch_sentences = sentences[i:i+BATCH_SIZE]
            response = client.embeddings.create(
                input=batch_sentences,
                model="text-embedding-3-large"
            )

            batch_embedding_matrix = np.zeros((len(batch_sentences), len(response.data[0].embedding)))

            # Populate the batch embedding matrix
            for j, embedding_obj in enumerate(response.data):
                embedding = np.array(embedding_obj.embedding)
                batch_embedding_matrix[j] = embedding

            # Append the batch embedding matrix to the main embedding matrix
            if embedding_matrix is None:
                embedding_matrix = batch_embedding_matrix
            else:
                embedding_matrix = np.concatenate((embedding_matrix, batch_embedding_matrix), axis=0)

        similarity_matrix = np.dot(embedding_matrix, embedding_matrix.T)

        return similarity_matrix

    def calculate_reward(self, matrix, start, end):
        sub_matrix = matrix[start:end+1, start:end+1]
        return np.sum(sub_matrix)

    # def calculate_local_density(self, matrix, i, window_size=3):
    #     n = matrix.shape[0]
    #     half_window = window_size // 2
    #     start_i = max(0, i - half_window)
    #     end_i = min(n, i + half_window + 1)
    #     local_area = matrix[start_i:end_i, start_i:end_i]
    #     return np.mean(local_area)

    def _optimal_segmentation(self, matrix, max_cluster_size, window_size=3):
        mean_value = np.mean(matrix[np.triu_indices(matrix.shape[0], k=1)])
        matrix = matrix - mean_value  # Normalize the matrix
        np.fill_diagonal(matrix, 0)  # Set diagonal to 1 to avoid trivial solutions

        n = matrix.shape[0]
        dp = np.zeros(n)
        segmentation = np.zeros(n, dtype=int)

        for i in range(n):
            for size in range(1, max_cluster_size + 1):
                if i - size + 1 >= 0:
                    # local_density = calculate_local_density(matrix, i, window_size)
                    reward = self.calculate_reward(matrix, i - size + 1, i)
                    # Adjust reward based on local density
                    adjusted_reward = reward
                    if i - size >= 0:
                        adjusted_reward += dp[i - size]
                    if adjusted_reward > dp[i]:
                        dp[i] = adjusted_reward
                        segmentation[i] = i - size + 1

        clusters = []
        i = n - 1
        while i >= 0:
            start = segmentation[i]
            clusters.append((start, i))
            i = start - 1

        clusters.reverse()
        return clusters
        

    def split_text(self, text):
        sentences = self.splitter.split_text(text)

        similarity_matrix = self._get_similarity_matrix(sentences)

        clusters = self._optimal_segmentation(similarity_matrix, max_cluster_size=self.max_cluster)

        docs = [' '.join(sentences[start:end+1]) for start, end in clusters]

        return docs