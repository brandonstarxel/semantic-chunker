from chroma_research import BaseChunker, GeneralBenchmark
from chroma_research.chunking import ClusterSemanticChunker
from chromadb.utils import embedding_functions

# Instantiate benchmark
benchmark = GeneralBenchmark()

# Choose embedding function
default_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="OPENAI_API_KEY",
    model_name="text-embedding-3-large"
)

# Instantiate chunker and run the benchmark
chunker = ClusterSemanticChunker(default_ef, max_chunk_size=400)
results = benchmark.run(chunker, default_ef)

print(results)
# {'iou_mean': 0.18255175232840098, 'iou_std': 0.12773219595465307, 
# 'recall_mean': 0.8973469551927365, 'recall_std': 0.29042203879923994}