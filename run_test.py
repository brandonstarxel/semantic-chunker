import os
from openai import OpenAI
from utils import get_retrieval_precision_prompt
import json
import numpy as np
import matplotlib.pyplot as plt
import backoff
import time

OPENAI_API_KEY = os.getenv('OPENAI_CHROMA_API_KEY')

client = OpenAI(api_key=OPENAI_API_KEY)

import chromadb.utils.embedding_functions as embedding_functions
import chromadb
import os

chroma_client = chromadb.PersistentClient(path="data/chroma_db")

OPENAI_API_KEY = os.getenv('OPENAI_CHROMA_API_KEY')

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY,
                model_name="text-embedding-ada-002"
            )

collection = chroma_client.get_collection("chuck_10", embedding_function=openai_ef)

# Setup backoff to retry on all exceptions
@backoff.on_exception(backoff.expo,
                      Exception,  # Broadly catching all exceptions
                      max_tries=8)
def get_retrieval_precision_indicator(question, context):
    try:
        completion = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": get_retrieval_precision_prompt(question, context)},
                {"role": "user", "content": "Is this CONTEXT relavent?"}
            ]
            )
        return 1 if completion.choices[0].message.content.lower().strip() == 'true' else 0
    except Exception as e:
        print(f"Error: {e}")
        raise e

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
def execute_threads(num_tasks):
    with ThreadPoolExecutor(max_workers=num_tasks) as executor:
        futures = [executor.submit(get_results_row) for _ in range(num_tasks)]
        return [future.result() for future in futures]

def main():
    results = execute_threads(4)
    print("First 4 threads")
    time.sleep(30)
    results += execute_threads(4)
    print(results)
    # Create a list of all trials
    all_trials = results

    # Transpose the list to get precision at each rank
    precision_at_each_rank = list(map(list, zip(*all_trials)))

    # Create labels
    labels = ['P@1', 'P@2', 'P@3', 'P@4', 'P@5']

    # Create the boxplot
    plt.boxplot(precision_at_each_rank, vert=False, labels=labels)
    plt.title('Boxplot of Precision@K for GPT-4-Turbo, Ada-002')
    plt.xlabel('Precision')
    plt.ylabel('Rank')
    plt.show()
    return results

if __name__ == "__main__":
    main()


# GPT-4-Turbo, Ada-002
# [[0.8411214953271028, 0.8130841121495327, 0.8193146417445484, 0.8107476635514018, 0.7943925233644858], [0.8504672897196262, 0.8364485981308412, 0.8317757009345795, 0.8107476635514018, 0.7831775700934579], [0.8317757009345794, 0.8364485981308412, 0.8286604361370715, 0.8107476635514018, 0.7906542056074765], [0.8504672897196262, 0.8271028037383178, 0.8286604361370717, 0.8271028037383178, 0.8056074766355139], [0.8317757009345794, 0.8271028037383178, 0.8286604361370717, 0.8107476635514018, 0.7887850467289718], [0.8504672897196262, 0.8317757009345794, 0.8317757009345794, 0.8200934579439252, 0.7981308411214953], [0.8504672897196262, 0.8271028037383178, 0.8286604361370717, 0.8130841121495327, 0.7869158878504673], [0.8317757009345794, 0.8271028037383178, 0.822429906542056, 0.8107476635514018, 0.7831775700934578]]