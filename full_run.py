import os
import json
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from utils import num_tokens_from_string

OPENAI_API_KEY = os.getenv('OPENAI_CHROMA_API_KEY')

chroma_client = chromadb.PersistentClient(path="data/chroma_db")

# This class is designed to rapidly create new collections for various chunking methods.
# It's done via a class so that it holds state in the case of an error mid-way through the process.
class CollectionWriter:
    def __init__(self, path="data/chroma_db"):
        self.chroma_client = chromadb.PersistentClient(path=path)
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY,
                model_name="text-embedding-3-large"
            )
        self.collection = None
        self.index = 0

    def trial_run(self, collection_name, chunking_function):
        self.write_to_collection("trial_run", chunking_function, 0, trial_run=True)

    def write_to_collection(self, collection_name, chunking_function, start_index=0, trial_run=False):
        if not trial_run:
            self.collection = self.chroma_client.get_or_create_collection(name=collection_name, embedding_function=self.openai_ef)
        
        total_tokens = 0

        with open('data/train.jsonl', 'r') as file:
            failed_indices = []
            for index, line in enumerate(file):
                if index < start_index:
                    continue
                if index in [8, 138, 209, 216, 368, 400]:
                    continue
                self.index = index
                data = json.loads(line)
                try:
                    documents = chunking_function(data['content'])
                except:
                    documents = []
                metadatas = [{"id": data['id'], "title": data['title']} for _ in range(len(documents))]
                ids = [data['id']+":"+str(i) for i in range(len(documents))]
                if not trial_run:
                    try:
                        self.collection.add(
                            documents=documents,
                            metadatas=metadatas,
                            ids=ids
                        )
                    except:
                        failed_indices.append(index)
                        print(f"Failed at index {index}")
                # print(ids[:5])
                # print(metadatas[:5])
                # print(documents[0])
                try:
                    num_tokens = sum([num_tokens_from_string(doc) for doc in documents])
                except:
                    num_tokens = 0
                print(f"{index} Added {len(documents)} documents. Current tokens: {num_tokens}")
                # print(f"{index} Added {len(documents)} documents.")
                total_tokens += num_tokens
            print(f"Failed indices: {failed_indices}")
        print(f"Total tokens: {total_tokens}")
        print(f"Price: ${round((total_tokens/1000000) * 0.13, 2)}")
        # print(f"Price: ${round((total_tokens/1000000) * 0.02, 2)}")
        return self.collection

from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter

splitter = RecursiveCharacterTextSplitter(
    # chunk_size=1024,
    # chunk_overlap=256,
    chunk_size=400,
    chunk_overlap=200,
    length_function=num_tokens_from_string
)

# splitter = TokenTextSplitter(
#     encoding_name="cl100k_base",
#     chunk_size=512,
#     chunk_overlap=50,
# )

collection_writer = CollectionWriter()

def chunking_function(content):
    return splitter.split_text(content)

# chunked_collection = collection_writer.trial_run("chuck_1", chunking_function)
chunked_collection = collection_writer.write_to_collection("chuck_12", chunking_function)