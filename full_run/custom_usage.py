from chroma_research import BaseChunker, GeneralBenchmark
from chromadb.utils import embedding_functions

# Define a custom chunking class
class CustomChunker(BaseChunker):
    def split_text(self, text):
        # Custom chunking logic
        return [text[i:i+1200] for i in range(0, len(text), 1200)]

# Instantiate the custom chunker and benchmark
chunker = CustomChunker()
benchmark = GeneralBenchmark()

import os
OPENAI_API_KEY = os.getenv('OPENAI_CHROMA_API_KEY')

# Choose embedding function
default_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-large"
)

# Run the benchmark
results = benchmark.run(chunker, default_ef)

print(results)
# {'iou_mean': 0.17715979570301696, 'iou_std': 0.10619791407460026, 
#  'recall_mean': 0.7193555455030595, 'recall_std': 0.4291027882174142}
