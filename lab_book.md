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
