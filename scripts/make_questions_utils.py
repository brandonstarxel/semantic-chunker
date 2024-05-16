import pickle
# with open('./notebooks/new_questions_references.pkl', 'rb') as f:
#     questions_references = pickle.load(f)

from openai import OpenAI
import os
import json
import random

from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import re

OPENAI_API_KEY = os.getenv('OPENAI_CHROMA_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

question_maker_prompt = """
You are an agent that generates questions from provided text. Your job is to generate a question and provide the relevant sections from the text as references.

Instructions:
1. For each provided text, generate a question that can be answered solely by the facts in the text.
2. Extract all significant facts that answer the generated question.
3. Format the response in JSON format with two fields:
   - 'question': A question directly related to these facts, ensuring it can only be answered using the references provided.
   - 'references': A list of all text sections that answer the generated question. These must be exact copies from the original text and should be whole sentences where possible.

Notes: 
Make the question more specific.
Do not ask a question about multiple topics. 
Do not ask a question with over 5 references.

Example:

Text: "Experiment A: The temperature control test showed that at higher temperatures, the reaction rate increased significantly, resulting in quicker product formation. However, at extremely high temperatures, the reaction yield decreased due to the degradation of reactants.

Experiment B: The pH sensitivity test revealed that the reaction is highly dependent on acidity, with optimal results at a pH of 7. Deviating from this pH level in either direction led to a substantial drop in yield.

Experiment C: In the enzyme activity assay, it was found that the presence of a specific enzyme accelerated the reaction by a factor of 3. The absence of the enzyme, however, led to a sluggish reaction with an extended completion time.

Experiment D: The light exposure trial demonstrated that UV light stimulated the reaction, making it complete in half the time compared to the absence of light. Conversely, prolonged light exposure led to unwanted side reactions that contaminated the final product."

Response: {
  'oath': "I will not use the word 'and' in the question unless it is part of a proper noun. I will also make sure the question is concise.",
  'question': 'What experiments were done in this paper?',
  'references': ['Experiment A: The temperature control test showed that at higher temperatures, the reaction rate increased significantly, resulting in quicker product formation.', 'Experiment B: The pH sensitivity test revealed that the reaction is highly dependent on acidity, with optimal results at a pH of 7.', 'Experiment C: In the enzyme activity assay, it was found that the presence of a specific enzyme accelerated the reaction by a factor of 3.', 'Experiment D: The light exposure trial demonstrated that UV light stimulated the reaction, making it complete in half the time compared to the absence of light.']
}

DO NOT USE THE WORD 'and' IN THE QUESTION UNLESS IT IS PART OF A PROPER NOUN. YOU MUST INCLUDE THE OATH ABOVE IN YOUR RESPONSE.
YOU MUST ALSO NOT REPEAT A QUESTION THAT HAS ALREADY BEEN USED.
"""

# def normalize_text(input_text: str) -> str:
#     """
#     Normalize the input text by replacing special characters with standard ones.
    
#     Args:
#         input_text (str): The text to normalize.
        
#     Returns:
#         str: The normalized text.
#     """
#     # Define mappings from special characters to standard ones
#     replacements = {
#         '‘': "'", '’': "'", '“': '"', '”': '"', '″': '"', '′': "'",
#         '–': '-', '—': '-', '…': '...', '«': '"', '»': '"', "\n": ""
#     }
    
#     # Replace each character in the input text based on the mappings
#     for special, standard in replacements.items():
#         input_text = input_text.replace(special, standard)
    
#     return input_text

def find_query_despite_whitespace(document, query):

    # Normalize spaces and newlines in the query
    normalized_query = re.sub(r'\s+', ' ', query).strip()
    
    # Create a regex pattern from the normalized query to match any whitespace characters between words
    pattern = r'\s*'.join(re.escape(word) for word in normalized_query.split())
    
    # Compile the regex to ignore case and search for it in the document
    regex = re.compile(pattern, re.IGNORECASE)
    match = regex.search(document)
    
    if match:
        return document[match.start(): match.end()], match.start(), match.end()
    else:
        return None

def find_target_in_document(document, target):

    if target.endswith('.'):
        target = target[:-1]
    
    if target in document:
        start_index = document.find(target)
        end_index = start_index + len(target)
        return target, start_index, end_index
    else:
        raw_search = find_query_despite_whitespace(document, target)
        if raw_search is not None:
            return raw_search

    # Split the text into sentences
    sentences = re.split(r'[.!?]\s*|\n', document)

    # Find the sentence that matches the query best
    best_match = process.extractOne(target, sentences, scorer=fuzz.token_sort_ratio)

    if best_match[1] < 98:
        return None
    
    reference = best_match[0]

    start_index = document.find(reference)
    end_index = start_index + len(reference)

    return reference, start_index, end_index

def get_sub_string(document, start_index, end_index):
    return document[start_index:end_index]

def tuple_to_df_row(question_tuple, corpus_id):
    question, references = question_tuple
    references = [{'content': ref[0], 'start_index': ref[1], 'end_index': ref[2]} for ref in references]
    return {
        'question': question,
        'references': json.dumps(references),
        'corpus_id': corpus_id
    }

# 1800 characters because this is roughly under 400 tokens
def extract_question_and_references(corpus, text_description, document_length=4000, prev_questions=[]):
    if len(corpus) > document_length:
        start_index = random.randint(0, len(corpus) - document_length)
        document = corpus[start_index : start_index + document_length]
    else:
        document = corpus
    
    if prev_questions is not None:
        if len(prev_questions) > 20:
            questions_sample = random.sample(prev_questions, 20)
            prev_questions_str = '\n'.join(questions_sample)
        else:
            prev_questions_str = '\n'.join(prev_questions)
    else:
        prev_questions_str = ""

    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        response_format={ "type": "json_object" },
        max_tokens=600,
        messages=[
            {"role": "system", "content": question_maker_prompt},
            {"role": "user", "content": f"Text Description:{text_description}\n\nText: {document}\n\nThe following questions have already been used. Do not repeat them:{prev_questions_str}\n\n Do not repeat the above questions. Make your next question unique. Respond with references and a question in JSON. DO NOT USE THE WORD 'and' IN THE QUESTION UNLESS IT IS PART OF A PROPER NOUN."}
        ]
    )
    
    json_response = json.loads(completion.choices[0].message.content)

    # print(json_response)
    
    try:
        text_references = json_response['references']
    except KeyError:
        raise ValueError("The response does not contain a 'references' field.")
    try:
        question = json_response['question']
    except KeyError:
        raise ValueError("The response does not contain a 'question' field.")

    references = []
    for reference in text_references:
        target = find_target_in_document(corpus, reference)
        if target is not None:
            reference, start_index, end_index = target
            references.append((reference, start_index, end_index))
        else:
            raise ValueError(f"No match found in the document for the given reference.\nReference: {reference}")
    
    return question, references

import json
import random
import pandas as pd

# for i in range(100):
def generate_questions(corpus_id, corpus_desc):
    corpus = None
    with open(f'./data/{corpus_id}.md', 'r') as file:
        corpus = file.read()

    questions_df = pd.read_csv('./data/questions_df.csv')

    i = -1
    while True:
        i += 1
        while True:
            try:
                print(f"Trying Question {i}")
                questions_list = questions_df[questions_df['corpus_id'] == corpus_id]['question'].tolist()
                question, references = extract_question_and_references(corpus, corpus_desc, 4000, questions_list)
                if len(references) > 5:
                    raise ValueError("The number of references exceeds 5.")
                
                new_question = tuple_to_df_row((question, references), corpus_id)
                new_df = pd.DataFrame([new_question])
                questions_df = pd.concat([questions_df, new_df], ignore_index=True)
                questions_df.to_csv('./data/questions_df.csv', index=False)

                break
            except (ValueError, json.JSONDecodeError) as e:
                print(f"Error occurred: {e}")
                continue