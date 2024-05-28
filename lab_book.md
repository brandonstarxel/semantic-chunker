# Retrieval Precision Tests
Retrieval Precision measures the total number of relevant documents divided by the number of returned documents for a given query. There a few variables to keep track as we measure this metric.

Firstly, this metric is not a number but a distribution. This is because the temperature of the LLM evaluators give different results for the same run. Our early tests only ran once but later ran multiple times to get this distribution. 

## Variables:
Chunking method: [RecursiveTextSplitter or TokenTextSplitter]
Chunk Size: 200 - 1000 
Chunk Overlap: 0 - 200
Embedding Model: [text-embedding-3-small, text-embedding-3-large, BERT, etc]
LLM Eval Model: [GPT-3.5-Turbo, GPT-4-Turbo]

The final variable is K, the number of documents we count. K=1 is just the top document while K=5 is all 5 returned documents. We will use P@K to define this. 

## Best Result:
The best result was achieved with:




(ARAGOG's setup)
TokenTextSplitter
chunk_size=512
chunk_overlap=50
text-embedding-3-large
Score: 0.869

(OpenAI's setup)
RecursiveCharacterTextSplitter 
chunk_size=400
chunk_overlap=200
embedding-3-small
Score: 0.900

TokenTextSplitter
chunk_size=400
chunk_overlap=200
embedding-3-small
Score: 0.879

RecursiveCharacterTextSplitter
chunk_size=400
chunk_overlap=0
embedding-3-small
Score: 0.857

TokenTextSplitter
chunk_size=400
chunk_overlap=0
embedding-3-small
Score: 0.807

Here's a table that organizes the data you provided, grouping them by their setups and specifying their respective scores:

| Setup                     | Text Splitter                 | Chunk Size | Chunk Overlap | Embedding          | P@3  | P@1 |
|---------------------------|-------------------------------|------------|---------------|--------------------|--------|--------|
| ARAGOG            | TokenTextSplitter             | 512        | 50            | text-embedding-3-large | 0.869  | 0.916 |
| None            | TokenTextSplitter             | 512        | 50            | text-embedding-ada-002 | 0.822  | 0.822
| OpenAI        | RecursiveCharacterTextSplitter | 400        | 200           | embedding-3-large   | 0.906  | 0.935 |
| None        | RecursiveCharacterTextSplitter | 400        | 200           | embedding-3-small   | 0.900  | 0.916 |
| None        | TokenTextSplitter             | 400        | 200           | embedding-3-small   | 0.879  | 0.925 |
| None        | RecursiveCharacterTextSplitter | 400        | 0             | embedding-3-small   | 0.857  | 0.907 |
| None       | TokenTextSplitter             | 400        | 0             | embedding-3-small   | 0.807  | 0.888 |

New IoC scores and Recall Scores:

| Setup                     | Text Splitter                 | Chunk Size | Chunk Overlap | Embedding          | IoC  | Recall |
|---------------------------|-------------------------------|------------|---------------|--------------------|--------|--------|
| ARAGOG            | TokenTextSplitter             | 512        | 50            | text-embedding-3-large | 0.170 ± 0.100  | 0.893 ± 0.280 |
| OpenAI        | RecursiveCharacterTextSplitter | 400        | 200           | embedding-3-large   | 0.197 ± 0.096  | 0.924 ± 0.222 |
| None        | TokenTextSplitter | 400        | 200           | embedding-3-large   | 0.177 ± 0.093  | 0.883 ± 0.243 |
| None        | RecursiveCharacterTextSplitter | 400        | 0             | embedding-3-large   | 0.235 ± 0.113 | 0.901 ± 0.229 |
| None       | TokenTextSplitter             | 400        | 0             | embedding-3-large   | 0.226 ± 0.112  | 0.874 ± 0.252 |

| Setup                     | Text Splitter                 | Chunk Size | Chunk Overlap | Embedding          | IoC  | Recall |
|---------------------------|-------------------------------|------------|---------------|--------------------|--------|--------|
| ARAGOG | TokenTextSplitter | 512 | 50 | text-embedding-3-large | 0.144 ± 0.089 | 0.937 ± 0.241 |
| OpenAI | RecursiveCharacterTextSplitter | 400 | 200 | text-embedding-3-large | 0.154 ± 0.090 | 0.950 ± 0.211 |
| None | TokenTextSplitter | 400 | 200 | text-embedding-3-large | 0.137 ± 0.066 | 0.965 ± 0.161 |
| None | RecursiveCharacterTextSplitter | 400 | 0 | text-embedding-3-large | 0.203 ± 0.114 | 0.961 ± 0.181 |
| None | TokenTextSplitter | 400 | 0 | text-embedding-3-large | 0.182 ± 0.096 | 0.967 ± 0.158 |


| Setup                     | Text Splitter                 | Chunk Size | Chunk Overlap | Embedding          | IoC  | Recall |
|---------------------------|-------------------------------|------------|---------------|--------------------|--------|--------|
| ARAGOG | TokenTextSplitter | 512 | 50 | text-embedding-3-large | 0.121 ± 0.089 | 0.947 ± 0.221 |
| OpenAI | RecursiveCharacterTextSplitter | 400 | 200 | text-embedding-3-large | 0.129 ± 0.090 | 0.956 ± 0.199 |
| None | TokenTextSplitter | 400 | 200 | text-embedding-3-large | 0.115 ± 0.070 | 0.967 ± 0.160 |
| None | RecursiveCharacterTextSplitter | 400 | 0 | text-embedding-3-large | 0.169 ± 0.117 | 0.953 ± 0.204 |
| None | TokenTextSplitter | 400 | 0 | text-embedding-3-large | 0.155 ± 0.098 | 0.975 ± 0.139 |


| Setup                     | Text Splitter                 | Chunk Size | Chunk Overlap | Embedding          | IoU  | Recall | Brute IoU | Brute Recall |
|---------------------------|-------------------------------|------------|---------------|--------------------|--------|--------|-----|-----|
| ARAGOG | TokenTextSplitter | 512 | 50 | text-embedding-3-large | 0.085 ± 0.058 | 0.998 ± 0.020 | 0.083 ± 0.056 | 1.000 ± 0.000 |
| OpenAI | RecursiveCharacterTextSplitter | 400 | 200 | text-embedding-3-large | 0.083 ± 0.051 | 0.990 ± 0.101 | 0.083 ± 0.054 | 1.000 ± 0.000 |
| None | TokenTextSplitter | 400 | 200 | text-embedding-3-large | 0.078 ± 0.052 | 0.986 ± 0.106 | 0.077 ± 0.051 | 1.000 ± 0.000 |
| None | RecursiveCharacterTextSplitter | 400 | 0 | text-embedding-3-large | 0.118 ± 0.076 | 0.986 ± 0.106 | 0.116 ± 0.071 | 1.000 ± 0.001 |
| None | TokenTextSplitter | 400 | 0 | text-embedding-3-large | 0.111 ± 0.079 | 0.990 ± 0.101 | 0.112 ± 0.078 | 1.000 ± 0.000 |

| Setup                     | Text Splitter                 | Chunk Size | Chunk Overlap | Embedding          | IoU  | Recall | Brute IoU | Brute Recall |
|---------------------------|-------------------------------|------------|---------------|--------------------|--------|--------|-----|-----|
| ARAGOG | TokenTextSplitter | 512 | 50 | text-embedding-3-large | 0.089 ± 0.058 | 0.950 ± 0.210 | 0.092 ± 0.056 | 1.000 ± 0.000 |
| OpenAI | RecursiveCharacterTextSplitter | 400 | 200 | text-embedding-3-large | 0.135 ± 0.109 | 0.958 ± 0.195 | 0.133 ± 0.100 | 1.000 ± 0.001 |
| None | TokenTextSplitter | 400 | 200 | text-embedding-3-large | 0.088 ± 0.053 | 0.954 ± 0.193 | 0.082 ± 0.044 | 1.000 ± 0.000 |
| None | RecursiveCharacterTextSplitter | 400 | 0 | text-embedding-3-large | 0.155 ± 0.112 | 0.945 ± 0.223 | 0.162 ± 0.107 | 1.000 ± 0.001 |
| None | TokenTextSplitter | 400 | 0 | text-embedding-3-large | 0.118 ± 0.073 | 0.969 ± 0.165 | 0.121 ± 0.071 | 1.000 ± 0.000 |


| Setup                     | Text Splitter                 | Chunk Size | Chunk Overlap | Embedding          | IoU  | Recall | Brute IoU | Brute Recall |
|---------------------------|-------------------------------|------------|---------------|--------------------|--------|--------|-----|-----|
| ARAGOG | TokenTextSplitter | 512 | 50 | text-embedding-3-large | 0.090 ± 0.069 | 0.881 ± 0.314 | 0.097 ± 0.063 | 1.000 ± 0.000 |
| OpenAI | RecursiveCharacterTextSplitter | 400 | 200 | text-embedding-3-large | 0.129 ± 0.116 | 0.853 ± 0.350 | 0.142 ± 0.106 | 1.000 ± 0.001 |
| None | TokenTextSplitter | 400 | 200 | text-embedding-3-large | 0.079 ± 0.071 | 0.766 ± 0.410 | 0.087 ± 0.051 | 1.000 ± 0.000 |
| None | RecursiveCharacterTextSplitter | 400 | 0 | text-embedding-3-large | 0.151 ± 0.138 | 0.831 ± 0.368 | 0.175 ± 0.127 | 1.000 ± 0.001 |
| None | TokenTextSplitter | 400 | 0 | text-embedding-3-large | 0.117 ± 0.087 | 0.885 ± 0.306 | 0.128 ± 0.080 | 1.000 ± 0.000 |



| Setup                     | Text Splitter                 | Chunk Size | Chunk Overlap | Embedding          | IoU  | Recall | Brute IoU | Brute Recall |
|---------------------------|-------------------------------|------------|---------------|--------------------|--------|--------|-----|-----|
| ARAGOG | TokenTextSplitter | 512 | 50 | text-embedding-3-large | 0.086 ± 0.071 | 0.840 ± 0.359 | 0.098 ± 0.063 | 1.000 ± 0.000 |
| OpenAI | RecursiveCharacterTextSplitter | 400 | 200 | text-embedding-3-large | 0.128 ± 0.118 | 0.828 ± 0.373 | 0.143 ± 0.105 | 1.000 ± 0.001 |
| None | TokenTextSplitter | 400 | 200 | text-embedding-3-large | 0.088 ± 0.071 | 0.824 ± 0.367 | 0.087 ± 0.051 | 1.000 ± 0.000 |
| None | RecursiveCharacterTextSplitter | 400 | 0 | text-embedding-3-large | 0.155 ± 0.143 | 0.831 ± 0.367 | 0.178 ± 0.131 | 1.000 ± 0.001 |
| None | TokenTextSplitter | 400 | 0 | text-embedding-3-large | 0.115 ± 0.089 | 0.856 ± 0.338 | 0.129 ± 0.081 | 1.000 ± 0.000 |
| LangChain | SemanticChunker | 0 | 0 | text-embedding-3-large | 0.073 ± 0.104 | 0.824 ± 0.379 | 0.079 ± 0.097 | 0.998 ± 0.032 |


State of the Union Address:
| Setup                     | Text Splitter                 | Chunk Size | Chunk Overlap | Embedding          | IoU  | Recall | Brute IoU | Brute Recall |
|---------------------------|-------------------------------|------------|---------------|--------------------|--------|--------|-----|-----|
| ARAGOG | TokenTextSplitter | 512 | 50 | text-embedding-3-large | 0.075 ± 0.052 | 0.993 ± 0.064 | 0.074 ± 0.051 | 1.000 ± 0.000 |
| OpenAI | RecursiveCharacterTextSplitter | 400 | 200 | text-embedding-3-large | 0.076 ± 0.043 | 0.989 ± 0.102 | 0.075 ± 0.043 | 1.000 ± 0.000 |
| None | TokenTextSplitter | 400 | 200 | text-embedding-3-large | 0.068 ± 0.043 | 0.979 ± 0.144 | 0.067 ± 0.042 | 1.000 ± 0.000 |
| None | RecursiveCharacterTextSplitter | 400 | 0 | text-embedding-3-large | 0.108 ± 0.073 | 0.978 ± 0.143 | 0.110 ± 0.072 | 0.999 ± 0.002 |
| None | TokenTextSplitter | 400 | 0 | text-embedding-3-large | 0.099 ± 0.067 | 0.989 ± 0.102 | 0.100 ± 0.066 | 1.000 ± 0.000 |
| LangChain | SemanticChunker | 0 | 0 | text-embedding-3-large | 0.104 ± 0.134 | 0.987 ± 0.103 | 0.102 ± 0.130 | 0.999 ± 0.003 |


Wikitexts:
| Setup                     | Text Splitter                 | Chunk Size | Chunk Overlap | Embedding          | IoU  | Recall | Brute IoU | Brute Recall |
|---------------------------|-------------------------------|------------|---------------|--------------------|--------|--------|-----|-----|
| ARAGOG | TokenTextSplitter | 512 | 50 | text-embedding-3-large | 0.095 ± 0.060 | 0.928 ± 0.254 | 0.102 ± 0.056 | 1.000 ± 0.000 |
| OpenAI | RecursiveCharacterTextSplitter | 400 | 200 | text-embedding-3-large | 0.168 ± 0.115 | 0.943 ± 0.226 | 0.168 ± 0.104 | 1.000 ± 0.001 |
| None | TokenTextSplitter | 400 | 200 | text-embedding-3-large | 0.101 ± 0.055 | 0.943 ± 0.204 | 0.091 ± 0.043 | 1.000 ± 0.000 |
| None | RecursiveCharacterTextSplitter | 400 | 0 | text-embedding-3-large | 0.187 ± 0.123 | 0.938 ± 0.235 | 0.197 ± 0.117 | 0.999 ± 0.004 |
| None | TokenTextSplitter | 400 | 0 | text-embedding-3-large | 0.131 ± 0.074 | 0.953 ± 0.199 | 0.135 ± 0.070 | 1.000 ± 0.000 |
| LangChain | SemanticChunker | 0 | 0 | text-embedding-3-large | 0.077 ± 0.090 | 0.944 ± 0.226 | 0.080 ± 0.089 | 1.000 ± 0.002 |


Re-runs:
| Setup                     | Text Splitter                 | Chunk Size | Chunk Overlap | Embedding          | IoU  | Recall | Brute IoU | Brute Recall |
|---------------------------|-------------------------------|------------|---------------|--------------------|--------|--------|-----|-----|
| None | TokenTextSplitter | 400 | 0 | text-embedding-3-large | 0.115 ± 0.089 | 0.863 ± 0.331 | 0.129 ± 0.081 | 1.000 ± 0.000 |
| None | TokenTextSplitter | 400 | 0 | text-embedding-3-large | 0.117 ± 0.088 | 0.872 ± 0.320 | 0.129 ± 0.081 | 1.000 ± 0.000 |
| None | TokenTextSplitter | 400 | 0 | text-embedding-3-large | 0.115 ± 0.090 | 0.852 ± 0.341 | 0.129 ± 0.081 | 1.000 ± 0.000 |

| Setup                     | Text Splitter                 | Chunk Size | Chunk Overlap | Embedding          | IoU  | Recall | Brute IoU | Brute Recall |
|---------------------------|-------------------------------|------------|---------------|--------------------|--------|--------|-----|-----|
| ARAGOG | TokenTextSplitter | 512 | 50 | text-embedding-3-large | 0.083 ± 0.078 | 0.741 ± 0.418 | 0.098 ± 0.063 | 1.000 ± 0.000 |
| OpenAI | RecursiveCharacterTextSplitter | 400 | 200 | text-embedding-3-large | 0.124 ± 0.117 | 0.777 ± 0.409 | 0.143 ± 0.105 | 1.000 ± 0.001 |
| None | TokenTextSplitter | 400 | 200 | text-embedding-3-large | 0.094 ± 0.079 | 0.815 ± 0.377 | 0.087 ± 0.051 | 1.000 ± 0.000 |
| None | RecursiveCharacterTextSplitter | 400 | 0 | text-embedding-3-large | 0.138 ± 0.139 | 0.716 ± 0.438 | 0.178 ± 0.131 | 0.999 ± 0.003 |
| None | TokenTextSplitter | 400 | 0 | text-embedding-3-large | 0.103 ± 0.093 | 0.723 ± 0.429 | 0.129 ± 0.081 | 1.000 ± 0.000 |





| Setup                     | Text Splitter                 | Chunk Size | Chunk Overlap | Embedding          | IoU  | Recall | Brute IoU | Brute Recall |
|---------------------------|-------------------------------|------------|---------------|--------------------|--------|--------|-----|-----|
| ARAGOG | TokenTextSplitter | 512 | 50 | text-embedding-3-large | 0.086 ± 0.071 | 0.840 ± 0.359 | 0.098 ± 0.063 | 1.000 ± 0.000 |
| OpenAI | RecursiveCharacterTextSplitter | 400 | 200 | text-embedding-3-large | 0.128 ± 0.118 | 0.828 ± 0.373 | 0.143 ± 0.105 | 1.000 ± 0.001 |
| None | TokenTextSplitter | 400 | 200 | text-embedding-3-large | 0.088 ± 0.071 | 0.824 ± 0.367 | 0.087 ± 0.051 | 1.000 ± 0.000 |
| None | RecursiveCharacterTextSplitter | 400 | 0 | text-embedding-3-large | 0.155 ± 0.143 | 0.831 ± 0.367 | 0.178 ± 0.131 | 1.000 ± 0.001 |
| None | TokenTextSplitter | 400 | 0 | text-embedding-3-large | 0.115 ± 0.089 | 0.856 ± 0.338 | 0.129 ± 0.081 | 1.000 ± 0.000 |
| LangChain | SemanticChunker | 0 | 0 | text-embedding-3-large | 0.073 ± 0.104 | 0.824 ± 0.379 | 0.079 ± 0.097 | 0.998 ± 0.032 |
| ChromaDB | SemanticChunker | 400 (~242) | 0 | text-embedding-3-large | 0.165 ± 0.127 | 0.897 ± 0.290 | 0.182 ± 0.127 | 0.999 ± 0.004 |
| ChromaDB | SemanticChunker | 200 (~133) | 0 | text-embedding-3-large | 0.260 ± 0.195 | 0.815 ± 0.366 | 0.299 ± 0.174 | 0.999 ± 0.006 |
| LangChainXChroma | GregImprovedChunker | 300 | 0 | text-embedding-3-large | 0.123 ± 0.132 | 0.842 ± 0.349 | 0.135 ± 0.128 | 0.999 ± 0.002 |
| Pinecone | PineconeExampleChunker | 300 | 0 | text-embedding-3-large | 0.158 ± 0.138 | 0.853 ± 0.342 | 0.180 ± 0.130 | 1.000 ± 0.002 |
| AurelioLabs | StatisticalSemanticChunker | 300 (~258) | 0 | text-embedding-3-large | 0.186 ± 0.160 | 0.830 ± 0.364 | 0.216 ± 0.147 | 0.999 ± 0.003 |

Pinecone:
Mean Token Count: 333.8165137614679
Median Token Count: 292.0
Standard Deviation of Token Count: 197.72944316326928

AurelioLabs Chunker:
Mean Token Count: 277.6234096692112
Median Token Count: 258.0
Standard Deviation of Token Count: 176.3530674679


| None | RecursiveCharacterTextSplitter | 242 | 0 | text-embedding-3-large | 0.23225 ± 0.19710 | 0.82827 ± 0.36544 | 0.267 ± 0.177 | 0.999 ± 0.002 |
| None | TokenTextSplitter | 242 | 0 | text-embedding-3-large | 0.16270 ± 0.12712 | 0.80622 ± 0.37621 | 0.188 ± 0.109 | 0.999 ± 0.021 |



This is with max_chunk_size=400
We get the resulting stats. 
Mean: 242.19807121661722
Median: 284.5
Min: 1
Max: 401
Standard Deviation: 132.16643280603242


For max_chunk_size=200
Mean: 133.42991418062934
Median: 149.0
Min: 1
Max: 201
Standard Deviation: 62.99149769847911




| Corpus ID                     | Text Splitter                 | Chunk Size | Chunk Overlap | Embedding          | IoU  | Recall | Brute IoU | Brute Recall |
|---------------------------|-------------------------------|------------|---------------|--------------------|--------|--------|-----|-----|
| Wikitexts | GreggImprovedChunker | 300 | 0 | text-embedding-3-large | 0.013 ± 0.072 | 0.029 ± 0.154 | 0.542 ± 0.207 | 0.996 ± 0.005 |
| State of the union | GreggImprovedChunker | 300 | 0 | text-embedding-3-large | 0.464 ± 0.237 | 0.938 ± 0.171 | 0.489 ± 0.228 | 0.995 ± 0.010 |
| Chatlogs | GreggImprovedChunker | 300 | 0 | text-embedding-3-large | 0.016 ± 0.078 | 0.029 ± 0.139 | 0.486 ± 0.161 | 0.996 ± 0.002 |
| Finance | GreggImprovedChunker | 300 | 0 | text-embedding-3-large | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.427 ± 0.239 | 0.998 ± 0.003 |
| Pubmed | GreggImprovedChunker | 300 | 0 | text-embedding-3-large | 0.003 ± 0.037 | 0.009 ± 0.094 | 0.498 ± 0.231 | 0.997 ± 0.007 |
| All | GreggImprovedChunker | 300 | 0 | text-embedding-3-large | 0.379 ± 0.269 | 0.707 ± 0.383 | 0.496 ± 0.220 | 0.997 ± 0.006 |