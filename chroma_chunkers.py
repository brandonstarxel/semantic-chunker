from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import num_tokens_from_string, harsh_doc_search
import os
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from semantic_router.encoders import OpenAIEncoder
from semantic_router.splitters import RollingWindowSplitter
from semantic_chunkers import StatisticalChunker

class ClusterChunker():
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
    

class GregImprovedChunker():
    def __init__(self, avg_chunk_size=400, min_chunk_size=50, BERT=False):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=min_chunk_size,
            chunk_overlap=0,
            length_function=num_tokens_from_string
            )
        
        self.avg_chunk_size = avg_chunk_size

    def combine_sentences(self, sentences, buffer_size=1):
        # Go through each sentence dict
        for i in range(len(sentences)):

            # Create a string that will hold the sentences which are joined
            combined_sentence = ''

            # Add sentences before the current one, based on the buffer size.
            for j in range(i - buffer_size, i):
                # Check if the index j is not negative (to avoid index out of range like on the first one)
                if j >= 0:
                    # Add the sentence at index j to the combined_sentence string
                    combined_sentence += sentences[j]['sentence'] + ' '

            # Add the current sentence
            combined_sentence += sentences[i]['sentence']

            # Add sentences after the current one, based on the buffer size
            for j in range(i + 1, i + 1 + buffer_size):
                # Check if the index j is within the range of the sentences list
                if j < len(sentences):
                    # Add the sentence at index j to the combined_sentence string
                    combined_sentence += ' ' + sentences[j]['sentence']

            # Then add the whole thing to your dict
            # Store the combined sentence in the current sentence dict
            sentences[i]['combined_sentence'] = combined_sentence

        return sentences

    def calculate_cosine_distances(self, sentences):
        distances = []
        for i in range(len(sentences) - 1):
            embedding_current = sentences[i]['combined_sentence_embedding']
            embedding_next = sentences[i + 1]['combined_sentence_embedding']
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
            
            # Convert to cosine distance
            distance = 1 - similarity

            # Append cosine distance to the list
            distances.append(distance)

            # Store distance in the dictionary
            sentences[i]['distance_to_next'] = distance

        # Optionally handle the last sentence
        # sentences[-1]['distance_to_next'] = None  # or a default value

        return distances, sentences

    def split_text(self, text):
        sentences_strips = self.splitter.split_text(text)

        sentences = [{'sentence': x, 'index' : i} for i, x in enumerate(sentences_strips)]

        sentences = self.combine_sentences(sentences, 3)

        combined_sentences = [x['combined_sentence'] for x in sentences]

        OPENAI_API_KEY = os.getenv('OPENAI_CHROMA_API_KEY')
        client = OpenAI(api_key=OPENAI_API_KEY)

        BATCH_SIZE = 400
        for i in range(0, len(combined_sentences), BATCH_SIZE):
            batch_sentences = combined_sentences[i:i+BATCH_SIZE]
            response = client.embeddings.create(
                input=batch_sentences,
                model="text-embedding-3-large"
            )

            for j, sentence in enumerate(sentences[i:i+BATCH_SIZE]):
                sentence['combined_sentence_embedding'] = response.data[j].embedding

        distances, sentences = self.calculate_cosine_distances(sentences)

        total_tokens = sum(num_tokens_from_string(sentence['sentence']) for sentence in sentences)
        avg_chunk_size = self.avg_chunk_size
        number_of_cuts = total_tokens // avg_chunk_size

        # Define threshold limits
        lower_limit = 0.0
        upper_limit = 1.0

        # Convert distances to numpy array
        distances_np = np.array(distances)

        # Binary search for threshold
        while upper_limit - lower_limit > 1e-6:
            threshold = (upper_limit + lower_limit) / 2.0
            num_points_above_threshold = np.sum(distances_np > threshold)
            
            if num_points_above_threshold > number_of_cuts:
                lower_limit = threshold
            else:
                upper_limit = threshold

        indices_above_thresh = [i for i, x in enumerate(distances) if x > threshold] 
        
        # Initialize the start index
        start_index = 0

        # Create a list to hold the grouped sentences
        chunks = []

        # Iterate through the breakpoints to slice the sentences
        for index in indices_above_thresh:
            # The end index is the current breakpoint
            end_index = index

            # Slice the sentence_dicts from the current start index to the end index
            group = sentences[start_index:end_index + 1]
            combined_text = ' '.join([d['sentence'] for d in group])
            chunks.append(combined_text)
            
            # Update the start index for the next group
            start_index = index + 1

        # The last group, if any sentences remain
        if start_index < len(sentences):
            combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
            chunks.append(combined_text)

        return chunks


class PineconeExampleChunker:
    def __init__(self, OPENAI_API_KEY):
        self.OPENAI_API_KEY = OPENAI_API_KEY

    def split_text(self, corpus):
        encoder = OpenAIEncoder(name="text-embedding-3-large", openai_api_key=self.OPENAI_API_KEY)

        splitter = RollingWindowSplitter(
            encoder=encoder,
            dynamic_threshold=True,
            min_split_tokens=100,
            max_split_tokens=500,
            window_size=2,
            plot_splits=True,  # set this to true to visualize chunking
            enable_statistics=True  # to print chunking stats
        )

        splits = splitter([corpus])

        extracted_texts = []
        for split in splits:
            ref, _, __ = harsh_doc_search(corpus, ' '.join([x for x in split.docs]))
            extracted_texts.append(ref)

        return extracted_texts
    

class AurelioLabsStatisticalChunker:
    def __init__(self, OPENAI_API_KEY):
        self.OPENAI_API_KEY = OPENAI_API_KEY

    def split_text(self, corpus):
        encoder = OpenAIEncoder(name="text-embedding-3-large", openai_api_key=self.OPENAI_API_KEY)

        chunker = StatisticalChunker(encoder=encoder)
        chunks = chunker(docs=[corpus])
        chunks = chunks[0]

        extracted_texts = []
        for chunk in chunks:
            ref, _, __ = harsh_doc_search(corpus, ' '.join([x for x in chunk.splits]))
            extracted_texts.append(ref)

        return extracted_texts
    
class LlamaTextChunker:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=50,
            chunk_overlap=0,
            length_function=num_tokens_from_string
            )

    def split_text(self, text):
        chunks_to_split_after = [2, 4, 7, 9, 12, 16, 18, 20, 22, 25, 27, 30, 34, 36, 38, 40, 42, 44, 50, 52, 54, 56, 60, 63, 67, 70, 72, 78, 89, 91, 94, 97, 100, 103, 106, 108, 110, 112, 116, 118, 121, 124, 126, 128, 131, 134, 137, 140, 142, 146, 152, 159, 161, 164, 168, 172, 176, 178, 180, 184, 186, 189, 192, 194, 196, 198, 200, 203, 206, 212, 214, 216, 219, 221, 223, 226, 228, 230, 231, 233, 236, 239, 242, 244, 246, 248, 251, 258, 266, 270, 272, 275, 279, 283, 288, 290, 292, 294, 297, 302, 304, 306, 308, 310, 312, 314, 317, 320, 321, 324, 326, 329, 331, 333, 335, 346, 349, 350, 352, 356, 358, 364, 366, 368, 370, 373, 376, 380, 384, 386, 388, 390, 392, 394, 397, 399, 401, 403, 405, 407, 408, 411, 414, 418, 422, 424, 427, 430, 433, 436, 438, 440, 442, 444, 446, 448, 450, 452, 454, 456, 457, 459, 461, 463, 466, 468, 470, 472, 474, 476, 478, 480, 482, 485, 489, 494, 496, 500, 505, 510, 512, 514, 516, 518, 520, 522, 524, 526, 528, 530, 532, 536, 540, 546, 552, 565, 567, 571, 575, 577, 579, 581, 584, 586, 588, 591, 595, 597, 598, 602, 608, 610, 613, 616, 619, 621, 628, 635, 638, 643, 648, 650, 653, 656, 659]
        chunks_to_split_after = [i - 1 for i in chunks_to_split_after]
        
        chunks = self.splitter.split_text(text)

        docs = []
        current_chunk = ''
        for i, chunk in enumerate(chunks):
            current_chunk += chunk + ' '
            if i in chunks_to_split_after:
                docs.append(current_chunk.strip())
                current_chunk = ''
        if current_chunk:
            docs.append(current_chunk.strip())

        return docs