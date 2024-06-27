from ioc_recall import IoCRecall
from utils import num_tokens_from_string
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from chroma_chunkers import ClusterChunker, GregImprovedChunker, PineconeExampleChunker, AurelioLabsStatisticalChunker, GPTTextChunker
import os

# text_splitter = SemanticChunker(OpenAIEmbeddings())


splitter_configurations = [
    # ("ARAGOG", TokenTextSplitter, 512, 50, "cl100k_base"),
    # ("OpenAI", RecursiveCharacterTextSplitter, 400, 200, "cl100k_base"),
    # ("None", TokenTextSplitter, 400, 200, "cl100k_base"),
    # ("None", RecursiveCharacterTextSplitter, 133, 0, "cl100k_base"),
    # ("None", TokenTextSplitter, 133, 0, "cl100k_base"),
    # ("LangChain", SemanticChunker, 0, 0, "cl100k_base")
    # ("None", ClusterChunker, 400, 0, "cl100k_base"),
    # ("None", ClusterChunker, 200, 0, "cl100k_base"),
    # ("None", GregImprovedChunker, 300, 0, "cl100k_base"),
    # ("None", PineconeExampleChunker, 300, 0, "cl100k_base"),
    # ("None", AurelioLabsStatisticalChunker, 300, 0, "cl100k_base"),
    # ("None", GPTTextChunker, 300, 0, "cl100k_base"),

    # ("None", TokenTextSplitter, 400, 0, "cl100k_base"),
    # ("None", TokenTextSplitter, 400, 0, "cl100k_base"),
    # ("None", TokenTextSplitter, 400, 0, "cl100k_base"),
    # ("None", TokenTextSplitter, 200, 0, "cl100k_base"),
    ("None", RecursiveCharacterTextSplitter, 200, 0, "cl100k_base"),
    # ("None", TokenTextSplitter, 100, 0, "cl100k_base"),
    # ("None", TokenTextSplitter, 50, 0, "cl100k_base"),
    # ("None", TokenTextSplitter, 25, 0, "cl100k_base"),
]

print("Warning: metadata will be incorrect if a chunk is repeated since we use .find() to find the start index. This isn't pratically an issue for chunks over 1000 characters.")

OPENAI_API_KEY = os.getenv('OPENAI_CHROMA_API_KEY')

ioc_recall = IoCRecall()
# ioc_recall = IoCRecall(corpus_list=['wikitexts'])

# for name, splitter_type, chunk_size, chunk_overlap, encoding_name in reversed(splitter_configurations):
for name, splitter_type, chunk_size, chunk_overlap, encoding_name in splitter_configurations:
    if splitter_type == TokenTextSplitter:
        splitter = splitter_type(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=encoding_name
        )
    elif splitter_type == RecursiveCharacterTextSplitter:
        splitter = splitter_type(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=num_tokens_from_string
        )
    # elif splitter_type == ClusterChunker:
    elif splitter_type == SemanticChunker:
        splitter = SemanticChunker(OpenAIEmbeddings())
        splitter = ClusterChunker(max_chunk_size=chunk_size)
    elif splitter_type == GregImprovedChunker:
        splitter = GregImprovedChunker(avg_chunk_size=chunk_size)
    elif splitter_type == PineconeExampleChunker:
        splitter = PineconeExampleChunker(OPENAI_API_KEY=OPENAI_API_KEY)
    elif splitter_type == AurelioLabsStatisticalChunker:
        splitter = AurelioLabsStatisticalChunker(OPENAI_API_KEY=OPENAI_API_KEY)
    elif splitter_type == GPTTextChunker:
        splitter = GPTTextChunker()

    # ioc_score, recall_score, brute_ioc_score, brute_recall_score = score_chunker(splitter)
    ioc_score, recall_score, brute_ioc_score, brute_recall_score = ioc_recall.score_chunker(splitter, BERT=False)

    # variability_test(splitter)

    print(f"| {name} | {splitter_type.__name__} | {chunk_size} | {chunk_overlap} | text-embedding-3-large | {ioc_score} | {recall_score} | {brute_ioc_score} | {brute_recall_score} |")

# | ARAGOG | TokenTextSplitter | 512 | 50 | text-embedding-3-large | 0.144 ± 0.089 | 0.937 ± 0.241 |
# | OpenAI | RecursiveCharacterTextSplitter | 400 | 200 | text-embedding-3-large | 0.154 ± 0.090 | 0.950 ± 0.211 |
# | None | TokenTextSplitter | 400 | 200 | text-embedding-3-large | 0.137 ± 0.066 | 0.965 ± 0.161 |
# | None | RecursiveCharacterTextSplitter | 400 | 0 | text-embedding-3-large | 0.203 ± 0.114 | 0.961 ± 0.181 |
# | None | TokenTextSplitter | 400 | 0 | text-embedding-3-large | 0.182 ± 0.096 | 0.967 ± 0.158 |




# | None | TokenTextSplitter | 400 | 0 | text-embedding-3-large | 0.115 ± 0.089 | 0.863 ± 0.331 | 0.129 ± 0.081 | 1.000 ± 0.000 |
# | None | TokenTextSplitter | 200 | 0 | text-embedding-3-large | 0.190 ± 0.146 | 0.809 ± 0.370 | 0.217 ± 0.122 | 1.000 ± 0.000 |


# 824 824
# | None | TokenTextSplitter | 400 | 0 | text-embedding-3-large | 0.117 ± 0.088 | 0.872 ± 0.320 | 0.129 ± 0.081 | 1.000 ± 0.000 |
# 1644 1644
# | None | TokenTextSplitter | 200 | 0 | text-embedding-3-large | 0.199 ± 0.142 | 0.850 ± 0.329 | 0.217 ± 0.122 | 1.000 ± 0.000 |
# 3285 3285


# | None | TokenTextSplitter | 400 | 0 | text-embedding-3-large | 0.115 ± 0.089 | 0.863 ± 0.331 | 0.129 ± 0.081 | 1.000 ± 0.000 |
# | None | TokenTextSplitter | 400 | 0 | text-embedding-3-large | 0.117 ± 0.088 | 0.872 ± 0.320 | 0.129 ± 0.081 | 1.000 ± 0.000 |
# | None | TokenTextSplitter | 400 | 0 | text-embedding-3-large | 0.115 ± 0.090 | 0.852 ± 0.341 | 0.129 ± 0.081 | 1.000 ± 0.000 |