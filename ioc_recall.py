import chromadb.utils.embedding_functions as embedding_functions
import chromadb
import os
import pickle
import numpy as np

corpus = None
with open('data/state_of_the_union.md', 'r') as file:
    corpus = file.read()

questions_references = None
with open('notebooks/new_questions_references.pkl', 'rb') as f:
    questions_references = pickle.load(f)

print(len(questions_references))

def sum_of_ranges(ranges):
    return sum(end - start for start, end in ranges)

def union_ranges(ranges):
    # Sort ranges based on the starting index
    sorted_ranges = sorted(ranges, key=lambda x: x[0])
    
    # Initialize with the first range
    merged_ranges = [sorted_ranges[0]]
    
    for current_start, current_end in sorted_ranges[1:]:
        last_start, last_end = merged_ranges[-1]
        
        # Check if the current range overlaps or is contiguous with the last range in the merged list
        if current_start <= last_end:
            # Merge the two ranges
            merged_ranges[-1] = (last_start, max(last_end, current_end))
        else:
            # No overlap, add the current range as new
            merged_ranges.append((current_start, current_end))
    
    return merged_ranges

def intersect_two_ranges(range1, range2):
    # Unpack the ranges
    start1, end1 = range1
    start2, end2 = range2
    
    # Calculate the maximum of the starting indices and the minimum of the ending indices
    intersect_start = max(start1, start2)
    intersect_end = min(end1, end2)
    
    # Check if the intersection is valid (the start is less than or equal to the end)
    if intersect_start <= intersect_end:
        return (intersect_start, intersect_end)
    else:
        return None  # Return an None if there is no intersection
    
# Define the difference function
def difference(ranges, target):
    """
    Takes a set of ranges and a target range, and returns the difference.
    
    Args:
    - ranges (list of tuples): A list of tuples representing ranges. Each tuple is (a, b) where a <= b.
    - target (tuple): A tuple representing a target range (c, d) where c <= d.
    
    Returns:
    - List of tuples representing ranges after removing the segments that overlap with the target range.
    """
    result = []
    target_start, target_end = target

    for start, end in ranges:
        if end < target_start or start > target_end:
            # No overlap
            result.append((start, end))
        elif start < target_start and end > target_end:
            # Target is a subset of this range, split it into two ranges
            result.append((start, target_start))
            result.append((target_end, end))
        elif start < target_start:
            # Overlap at the start
            result.append((start, target_start))
        elif end > target_end:
            # Overlap at the end
            result.append((target_end, end))
        # Else, this range is fully contained by the target, and is thus removed

    return result

def find_target_in_document(document, target):
    start_index = document.find(target)
    if start_index == -1:
        return None
    end_index = start_index + len(target)
    return start_index, end_index

def get_chunks_and_metadata(splitter, corpus):
    documents = splitter.split_text(corpus)
    metadatas = []
    for document in documents:
        start_index, end_index = find_target_in_document(corpus, document)
        metadatas.append({"start_index": start_index, "end_index": end_index})
    return documents, metadatas

def full_precision_score(questions_references, chunk_metadatas):
    ioc_scores = []
    recall_scores = []
    for question, references in questions_references:
        # Unpack question and references
        #  = question_references
        ioc_score = 0
        numerator_sets = []
        denominator_chunks_sets = []
        unused_highlights = [(x[1], x[2]) for x in references]

        for metadata in chunk_metadatas:
            # Unpack chunk start and end indices
            chunk_start, chunk_end = metadata['start_index'], metadata['end_index']
            
            for reference, ref_start, ref_end in references:
                # Calculate intersection between chunk and reference
                intersection = intersect_two_ranges((chunk_start, chunk_end), (ref_start, ref_end))
                
                if intersection is not None:
                    # Remove intersection from unused highlights
                    unused_highlights = difference(unused_highlights, intersection)

                    # Add intersection to numerator sets
                    numerator_sets = union_ranges([intersection] + numerator_sets)
                    
                    # Add chunk to denominator sets
                    denominator_chunks_sets = union_ranges([(chunk_start, chunk_end)] + denominator_chunks_sets)
        
        # Combine unused highlights and chunks for final denominator
        denominator_sets = union_ranges(denominator_chunks_sets + unused_highlights)
        
        # Calculate ioc_score if there are numerator sets
        if numerator_sets:
            ioc_score = sum_of_ranges(numerator_sets) / sum_of_ranges(denominator_sets)
        
        ioc_scores.append(ioc_score)

        recall_score = 1 - (sum_of_ranges(unused_highlights) / sum_of_ranges([(x[1], x[2]) for x in references]))
        recall_scores.append(recall_score)

    return ioc_scores, recall_scores

def scores_from_dataset_and_retrievals(questions_references, question_metadatas):
    ioc_scores = []
    recall_scores = []
    for question_references, metadatas in zip(questions_references, question_metadatas):
        # Unpack question and references
        question, references = question_references
        ioc_score = 0
        numerator_sets = []
        denominator_chunks_sets = []
        unused_highlights = [(x[1], x[2]) for x in references]

        for metadata in metadatas:
            # Unpack chunk start and end indices
            chunk_start, chunk_end = metadata['start_index'], metadata['end_index']
            
            for reference, ref_start, ref_end in references:
                # Calculate intersection between chunk and reference
                intersection = intersect_two_ranges((chunk_start, chunk_end), (ref_start, ref_end))
                
                if intersection is not None:
                    # Remove intersection from unused highlights
                    unused_highlights = difference(unused_highlights, intersection)

                    # Add intersection to numerator sets
                    numerator_sets = union_ranges([intersection] + numerator_sets)
                    
                    # Add chunk to denominator sets
                    denominator_chunks_sets = union_ranges([(chunk_start, chunk_end)] + denominator_chunks_sets)
        
        # Combine unused highlights and chunks for final denominator
        denominator_sets = union_ranges(denominator_chunks_sets + unused_highlights)
        
        # Calculate ioc_score if there are numerator sets
        if numerator_sets:
            ioc_score = sum_of_ranges(numerator_sets) / sum_of_ranges(denominator_sets)
        
        ioc_scores.append(ioc_score)

        recall_score = 1 - (sum_of_ranges(unused_highlights) / sum_of_ranges([(x[1], x[2]) for x in references]))
        recall_scores.append(recall_score)

    return ioc_scores, recall_scores

def chunker_to_collection(chunker):
    chroma_client = chromadb.PersistentClient(path="../data/chroma_db")
    OPENAI_API_KEY = os.getenv('OPENAI_CHROMA_API_KEY')
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=OPENAI_API_KEY,
                    model_name="text-embedding-3-large"
                )
    
    collection_name = "auto_chunk"
    
    try:
        chroma_client.delete_collection(collection_name)
    except ValueError as e:
        pass

    collection = chroma_client.create_collection(collection_name, embedding_function=openai_ef)

    docs, metas = get_chunks_and_metadata(chunker, corpus)

    collection.add(
        documents=docs,
        metadatas=metas,
        ids=[str(i) for i in range(len(docs))]
    )

    return collection

def score_chunker(chunker):
    # print("Starting Chunking")
    collection = chunker_to_collection(chunker)
    # print("Chunking Complete")

    # print("Starting Retrieval")
    retrievals = collection.query(query_texts=[x[0] for x in questions_references], n_results=5)
    # print("Retrieval Complete")

    ioc_scores, recall_scores = scores_from_dataset_and_retrievals(questions_references, retrievals['metadatas'])
    brute_ioc_scores, brute_recall_scores = full_precision_score(questions_references, collection.get()['metadatas'])

    ioc_mean = np.mean(ioc_scores)
    ioc_std = np.std(ioc_scores)
    ioc_text = f"{ioc_mean:.3f} ± {ioc_std:.3f}"

    brute_ioc_mean = np.mean(brute_ioc_scores)
    brute_ioc_std = np.std(brute_ioc_scores)
    brute_ioc_text = f"{brute_ioc_mean:.3f} ± {brute_ioc_std:.3f}"

    recall_mean = np.mean(recall_scores)
    recall_std = np.std(recall_scores)
    recall_text = f"{recall_mean:.3f} ± {recall_std:.3f}"

    brute_recall_mean = np.mean(brute_recall_scores)
    brute_recall_std = np.std(brute_recall_scores)
    brute_recall_text = f"{brute_recall_mean:.3f} ± {brute_recall_std:.3f}"

    return ioc_text, recall_text, brute_ioc_text, brute_recall_text