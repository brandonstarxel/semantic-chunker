from chroma_research import BaseChunker, GeneralBenchmark
# from chroma_research.chunking import ClusterSemanticChunker
from chroma_research.chunking import ClusterSemanticChunker, LLMSemanticChunker
from chromadb.utils import embedding_functions
from utils import count_non_pad_tokens, num_tokens_from_string
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

import os
OPENAI_API_KEY = os.getenv('OPENAI_CHROMA_API_KEY')

# # Choose embedding function
default_ef = embedding_functions.SentenceTransformerEmbeddingFunction()
# default_ef = embedding_functions.OpenAIEmbeddingFunction(api_key = OPENAI_API_KEY, model_name="text-embedding-3-large")
# chunker = ClusterSemanticChunker(default_ef, max_chunk_size=400, length_function=num_tokens_from_string)

from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter

chunkers = [
    RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=125, length_function=count_non_pad_tokens, separators = ["\n\n", "\n", ".", "?", "!", " ", ""]),
    TokenTextSplitter(chunk_size=278, chunk_overlap=139, encoding_name="cl100k_base"),
    # RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0, length_function=count_non_pad_tokens),
    # TokenTextSplitter(chunk_size=278, chunk_overlap=0, encoding_name="cl100k_base"),
    # RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0, length_function=count_non_pad_tokens),
    # TokenTextSplitter(chunk_size=222, chunk_overlap=0, encoding_name="cl100k_base"),
    # ClusterSemanticChunker(default_ef, max_chunk_size=250, length_function=count_non_pad_tokens),
    # ClusterSemanticChunker(default_ef, max_chunk_size=200, length_function=count_non_pad_tokens)
]

# recall_means = []
db_to_save_chunks = "/Users/brandon/Desktop/MonteIntelligence/AARB_chunks"
import numpy as np
# Define a function to run the benchmark and print the results
# def run_benchmark_and_print_results(chunker, benchmark, default_ef):

recalls = []

def run_benchmark_and_print_results(chunker, default_ef, retrieve):
    # Run the benchmark
    benchmark = GeneralBenchmark()
    results = benchmark.run(chunker, default_ef, retrieve=retrieve, db_to_save_chunks=db_to_save_chunks)
    del benchmark
    
    chunker_name = chunker.__class__.__name__ if hasattr(chunker, '__class__') else "N/A"
    chunk_size = chunker._chunk_size if hasattr(chunker, '_chunk_size') else "N/A"

    if chunk_size == 278:
        chunk_size = 250
    elif chunk_size == 222:
        chunk_size = 200

    if chunker.__class__.__name__ == "ClusterSemanticChunker":
        chunk_overlap = 0
    else:
        chunk_overlap = chunker._chunk_overlap if hasattr(chunker, '_chunk_overlap') else "N/A"

    if chunk_overlap == 139:
        chunk_overlap = 125

    score_rows = []
    
    corpora_scores = results['corpora_scores']
    for corpus_name, corpus_scores in corpora_scores.items():
        if corpus_name != "wikitexts":
            continue
        brute_iou_mean = np.mean(corpus_scores['brute_iou_scores'])
        brute_iou_std = np.std(corpus_scores['brute_iou_scores'])
        iou_scores = np.mean(corpus_scores['iou_scores'])
        iou_std = np.std(corpus_scores['iou_scores'])
        recall_scores = np.mean(corpus_scores['recall_scores'])
        # print(corpus_scores['recall_scores'])
        recalls.append(corpus_scores['recall_scores'])
        recall_std = np.std(corpus_scores['recall_scores'])
        precision_scores = np.mean(corpus_scores['precision_scores'])
        precision_std = np.std(corpus_scores['precision_scores'])

        print(f"{chunker_name} & {chunk_size} & {chunk_overlap} & {corpus_name} & {recall_scores:.3f} ± {recall_std:.3f} & {precision_scores:.3f} ± {precision_std:.3f} & {brute_iou_mean:.3f} ± {brute_iou_std:.3f} & {iou_scores:.3f} ± {iou_std:.3f}\\\\")
        score_rows.append([chunker_name, chunk_size, chunk_overlap, corpus_name, recall_scores, recall_std, precision_scores, precision_std, brute_iou_mean, brute_iou_std, iou_scores, iou_std, retrieve])
    print(f"{chunker_name} & {chunk_size} & {chunk_overlap} & ALL & {results['recall_mean']:.3f} ± {results['recall_std']:.3f} & {results['precision_mean']:.3f} ± {results['precision_std']:.3f} & {results['iou_full_mean']:.3f} ± {results['iou_full_std']:.3f} & {results['iou_mean']:.3f} ± {results['iou_std']:.3f}\\\\")
    return score_rows
    # Print the results
    # print(f"iou_full: {results['iou_full_mean']:.3f} ± {results['iou_full_std']:.3f}")
    # print(f"iou: {results['iou_mean']:.3f} ± {results['iou_std']:.3f}")
    # print(f"recall: {results['recall_mean']:.3f} ± {results['recall_std']:.3f}")
    # print(f"precision: {results['precision_mean']:.3f} ± {results['precision_std']:.3f}")
# RecursiveCharacterTextSplitter & 250 & 125 & ALL & 0.783 ± 0.395 & 0.049 ± 0.045 & 0.219 ± 0.149 & 0.049 ± 0.045\\
# TokenTextSplitter & 250 & 125 & ALL & 0.822 ± 0.364 & 0.035 ± 0.030 & 0.114 ± 0.066 & 0.035 ± 0.030\\
# Define the chunkers with the configurations shown above
import pandas as pd

results_list = []
# retrieves = [-1, 5, 10]
retrieves = [5]
# Run the benchmark and print the results for each chunker
for retrieve in retrieves:
    for chunker in chunkers:
        sub_result = run_benchmark_and_print_results(chunker, default_ef, retrieve)
        results_list.extend(sub_result)

# recalls_ndarry = np.array(recalls

print(np.subtract(recalls[1], recalls[0]))
print(np.subtract(recalls[1], recalls[0]).mean())
print(np.subtract(recalls[1], recalls[0]).sum())

# Assuming the array is `recalls`
import numpy as np
# Get the indexes of the largest values in the array
largest_value_indexes = np.argsort(np.subtract(recalls[1], recalls[0]))[::-1]
print(largest_value_indexes)