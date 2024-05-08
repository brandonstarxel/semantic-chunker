from ioc_recall import score_chunker
from utils import num_tokens_from_string
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter

splitter_configurations = [
    ("ARAGOG", TokenTextSplitter, 512, 50, "cl100k_base"),
    ("OpenAI", RecursiveCharacterTextSplitter, 400, 200, "cl100k_base"),
    ("None", TokenTextSplitter, 400, 200, "cl100k_base"),
    ("None", RecursiveCharacterTextSplitter, 400, 0, "cl100k_base"),
    ("None", TokenTextSplitter, 400, 0, "cl100k_base")
]

for name, splitter_type, chunk_size, chunk_overlap, encoding_name in splitter_configurations:
    if splitter_type == TokenTextSplitter:
        splitter = splitter_type(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=encoding_name
        )
    else:
        splitter = splitter_type(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=num_tokens_from_string
        )

    ioc_score, recall_score, brute_ioc_score, brute_recall_score = score_chunker(splitter)

    print(f"| {name} | {splitter_type.__name__} | {chunk_size} | {chunk_overlap} | text-embedding-3-large | {ioc_score} | {recall_score} | {brute_ioc_score} | {brute_recall_score} |")

# | ARAGOG | TokenTextSplitter | 512 | 50 | text-embedding-3-large | 0.144 ± 0.089 | 0.937 ± 0.241 |
# | OpenAI | RecursiveCharacterTextSplitter | 400 | 200 | text-embedding-3-large | 0.154 ± 0.090 | 0.950 ± 0.211 |
# | None | TokenTextSplitter | 400 | 200 | text-embedding-3-large | 0.137 ± 0.066 | 0.965 ± 0.161 |
# | None | RecursiveCharacterTextSplitter | 400 | 0 | text-embedding-3-large | 0.203 ± 0.114 | 0.961 ± 0.181 |
# | None | TokenTextSplitter | 400 | 0 | text-embedding-3-large | 0.182 ± 0.096 | 0.967 ± 0.158 |