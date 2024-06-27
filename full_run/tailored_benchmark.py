from chroma_research import SyntheticBenchmark

# Specify the corpora paths and output CSV file
corpora_paths = [
    './corpora/pride_and_prejudice.txt',
    './corpora/sherlock_holmes.txt',
    './corpora/the_picture_of_dorian_gray.txt',
    # '/Users/brandon/Desktop/chroma_research/chroma_research/benchmarking/general_benchmark_data/corpora/chatlogs.md',
    './corpora/fake_corpora.txt',
    # '/Users/brandon/Desktop/chroma_research/chroma_research/benchmarking/general_benchmark_data/corpora/finance.md',
    # '/Users/brandon/Desktop/chroma_research/chroma_research/benchmarking/general_benchmark_data/corpora/pubmed.md',
    # '/Users/brandon/Desktop/chroma_research/chroma_research/benchmarking/general_benchmark_data/corpora/state_of_the_union.md',
    # '/Users/brandon/Desktop/chroma_research/chroma_research/benchmarking/general_benchmark_data/corpora/wikitexts.md'
    # Add more corpora files as needed
]
questions_csv_path = "fake_questions.csv"
# questions_csv_path = "/Users/brandon/Desktop/chroma_research/chroma_research/benchmarking/general_benchmark_data/questions_df.csv"

import pandas as pd

# Load the questions CSV file
questions_df = pd.read_csv(questions_csv_path)

# Find the number of unique rows groupby 'corpus_id' and print
unique_rows = questions_df.groupby('corpus_id').nunique()
print(unique_rows)


import os
OPENAI_API_KEY = os.getenv('OPENAI_CHROMA_API_KEY')

# Instantiate the benchmark
synethetic_benchmark = SyntheticBenchmark(corpora_paths, questions_csv_path, openai_api_key=OPENAI_API_KEY)

# Generate questions and highlights
synethetic_benchmark.generate_questions_and_highlights(approximate_highlights=False)
# synethetic_benchmark.generate_questions_and_highlights(approximate_highlights=True)

# corpora_subset = ['state_of_the_union', 'finance', 'pubmed', 'wikitexts', 'chatlogs']
# corpora_subset = ['pubmed']

# synethetic_benchmark.filter_poor_highlights(threshold=0.4, corpora_subset=corpora_subset)

# synethetic_benchmark.filter_duplicates(threshold=0.67, corpora_subset=corpora_subset)

# from chroma_research import BaseChunker, GeneralBenchmark
# from chroma_research.chunking import ClusterSemanticChunker
# from chromadb.utils import embedding_functions

# # Define a custom chunking class
# class CustomChunker(BaseChunker):
#     def split_text(self, text):
#         # Custom chunking logic
#         # return [text[i:i+1200] for i in range(0, len(text), 1200)]
#         # return [text[0:201], text[201:]]
#         print("I AM THE SPLITTER: ", [text[0:130], text[131:]])
#         return [text[0:130], text[131:]]
    
# # {'iou_mean': 0.24971376811594204, 'iou_std': 0.08902577527377108, 'recall_mean': 0.6103405302223509, 'recall_std': 0.441175275435976} 
# chunker = CustomChunker()

# # Choose embedding function
# default_ef = embedding_functions.OpenAIEmbeddingFunction(
#     api_key=OPENAI_API_KEY,
#     model_name="text-embedding-3-large"
# )

# # chunker = ClusterSemanticChunker(embedding_function=default_ef)

# print("Running the benchmark...")

# # Run the benchmark
# results = synethetic_benchmark.run(chunker, default_ef)

# print(results)