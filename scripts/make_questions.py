import pickle
with open('./notebooks/new_questions_references.pkl', 'rb') as f:
    questions_references = pickle.load(f)

with open('./data/state_of_the_union.md', 'r') as file:
    data = file.read()

from openai import OpenAI
import os
import json
import random

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
  'question': 'What experiments were done in this paper?',
  'references': ['Experiment A: The temperature control test showed that at higher temperatures, the reaction rate increased significantly, resulting in quicker product formation.', 'Experiment B: The pH sensitivity test revealed that the reaction is highly dependent on acidity, with optimal results at a pH of 7.', 'Experiment C: In the enzyme activity assay, it was found that the presence of a specific enzyme accelerated the reaction by a factor of 3.', 'Experiment D: The light exposure trial demonstrated that UV light stimulated the reaction, making it complete in half the time compared to the absence of light.']
}
"""

def normalize_text(input_text: str) -> str:
    """
    Normalize the input text by replacing special characters with standard ones.
    
    Args:
        input_text (str): The text to normalize.
        
    Returns:
        str: The normalized text.
    """
    # Define mappings from special characters to standard ones
    replacements = {
        '‘': "'", '’': "'", '“': '"', '”': '"', '″': '"', '′': "'",
        '–': '-', '—': '-', '…': '...', '«': '"', '»': '"', "\n": ""
    }
    
    # Replace each character in the input text based on the mappings
    for special, standard in replacements.items():
        input_text = input_text.replace(special, standard)
    
    return input_text

def find_target_in_document(document, target):
    norm_corpus = normalize_text(document)
    norm_target = normalize_text(target)
    start_index = norm_corpus.find(norm_target)
    if start_index == -1:
        if norm_target[-1] == ".":
            start_index = norm_corpus.find(norm_target[:-1])
            if start_index == -1:
                return None
        else:
            return None
    start_index = start_index + document[:start_index].count("\n")
    end_index = start_index + len(target)
    return start_index, end_index

def get_sub_string(document, start_index, end_index):
    return document[start_index:end_index]

# 1800 characters because this is roughly under 400 tokens
# 1800 characters because this is roughly under 400 tokens
def extract_question_and_references(corpus, text_description, document_length=4000, prev_questions=[]):
    if len(corpus) > document_length:
        start_index = random.randint(0, len(corpus) - document_length)
        document = corpus[start_index : start_index + document_length]
    else:
        document = corpus
    questions_sample = random.sample(prev_questions, 50)
    prev_questions_str = '\n'.join(questions_sample)
    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        response_format={ "type": "json_object" },
        max_tokens=600,
        messages=[
            {"role": "system", "content": question_maker_prompt},
            {"role": "user", "content": f"Text Description:{text_description}\n\nText: {document}\n\nThe following questions have already been used. Do not repeat them:{prev_questions_str}Respond with references and a question in JSON."}
        ]
    )
    
    json_response = json.loads(completion.choices[0].message.content)
    
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
            start_index, end_index = target
            references.append((reference, start_index, end_index))
        else:
            raise ValueError(f"No match found in the document for the given reference.\nReference: {reference}")
    
    return question, references

import json
import random

# for i in range(100):
i = -1
while True:
    i += 1
    while True:
        try:
            print(f"Trying Question {i}")
            question, references = extract_question_and_references(data, "This is a transcribed segment from President Biden's State of the Union Address.", 4000, [q[0] for q in questions_references])
            if len(references) > 5:
                raise ValueError("The number of references exceeds 5.")
            questions_references.append((question, references))
            with open('./notebooks/new_questions_references.pkl', 'wb') as f:
                pickle.dump(questions_references, f)
            break
        except (ValueError, json.JSONDecodeError) as e:
            print(f"Error occurred: {e}")
            continue