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


| Setup                     | Text Splitter                 | Chunk Size | Chunk Overlap | Embedding          | IoC  | Recall | Brute IoC | Brute Recall |
|---------------------------|-------------------------------|------------|---------------|--------------------|--------|--------|-----|-----|
| ARAGOG | TokenTextSplitter | 512 | 50 | text-embedding-3-large | 0.085 ± 0.058 | 0.998 ± 0.020 | 0.083 ± 0.056 | 1.000 ± 0.000 |
| OpenAI | RecursiveCharacterTextSplitter | 400 | 200 | text-embedding-3-large | 0.083 ± 0.051 | 0.990 ± 0.101 | 0.083 ± 0.054 | 1.000 ± 0.000 |
| None | TokenTextSplitter | 400 | 200 | text-embedding-3-large | 0.078 ± 0.052 | 0.986 ± 0.106 | 0.077 ± 0.051 | 1.000 ± 0.000 |
| None | RecursiveCharacterTextSplitter | 400 | 0 | text-embedding-3-large | 0.118 ± 0.076 | 0.986 ± 0.106 | 0.116 ± 0.071 | 1.000 ± 0.001 |
| None | TokenTextSplitter | 400 | 0 | text-embedding-3-large | 0.111 ± 0.079 | 0.990 ± 0.101 | 0.112 ± 0.078 | 1.000 ± 0.000 |