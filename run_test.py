import os
from openai import OpenAI
from utils import get_retrieval_precision_prompt
import json
import numpy as np
import matplotlib.pyplot as plt

OPENAI_API_KEY = os.getenv('OPENAI_CHROMA_API_KEY')

client = OpenAI(api_key=OPENAI_API_KEY)

import chromadb.utils.embedding_functions as embedding_functions
import chromadb
import os

chroma_client = chromadb.PersistentClient(path="data/chroma_db")

OPENAI_API_KEY = os.getenv('OPENAI_CHROMA_API_KEY')

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY,
                model_name="text-embedding-3-large"
            )

collection = chroma_client.get_collection("chuck_8", embedding_function=openai_ef)

def get_retrieval_precision_indicator(question, context):
    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": get_retrieval_precision_prompt(question, context)},
            {"role": "user", "content": "Is this CONTEXT relavent?"}
        ]
        )
    
    return 1 if completion.choices[0].message.content.lower().strip() == 'true' else 0

with open('eval_questions/eval_data.json') as f:
    eval_data = json.load(f)

results = collection.query(query_texts=eval_data['questions'], n_results=5)

def get_results_row():
    retrieval_precision_matrix = np.zeros((len(eval_data['questions']), 5))
    for i, question in enumerate(eval_data['questions']):
        for j, context in enumerate(results['documents'][i]):
            retrieval_precision_matrix[i][j] = get_retrieval_precision_indicator(question, context)
        print(f"Question {i}, Context {j}: {retrieval_precision_matrix[i][j]}")
    precision_at_1 = np.mean(retrieval_precision_matrix[:, 0])
    precision_at_2 = np.mean(np.sum(retrieval_precision_matrix[:, :2], axis=1) / 2)
    precision_at_3 = np.mean(np.sum(retrieval_precision_matrix[:, :3], axis=1) / 3)
    precision_at_4 = np.mean(np.sum(retrieval_precision_matrix[:, :4], axis=1) / 4)
    precision_at_5 = np.mean(np.sum(retrieval_precision_matrix[:, :5], axis=1) / 5)
    return [precision_at_1, precision_at_2, precision_at_3, precision_at_4, precision_at_5]

from concurrent.futures import ThreadPoolExecutor

def main():
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(get_results_row) for _ in range(8)]
        results = [future.result() for future in futures]
    print(results)
    # Create a list of all trials
    all_trials = results

    # Transpose the list to get precision at each rank
    precision_at_each_rank = list(map(list, zip(*all_trials)))

    # Create labels
    labels = ['P@1', 'P@2', 'P@3', 'P@4', 'P@5']

    # Create the boxplot
    plt.boxplot(precision_at_each_rank, vert=False, labels=labels)
    plt.title('Boxplot of Precision@K for GPT-4-Turbo')
    plt.xlabel('Precision')
    plt.ylabel('Rank')
    plt.show()
    return results

if __name__ == "__main__":
    main()