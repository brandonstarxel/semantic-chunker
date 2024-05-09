from make_questions_utils import generate_questions

# generate_questions(corpus_id='state_of_the_union', corpus_desc="This is a transcribed segment from President Biden's State of the Union Address.")
generate_questions(corpus_id='wikitexts', corpus_desc="This is a section of Wikipedia articles.")
# with open(f'./data/wikitexts.md', 'r') as file:
#     corpus = file.read()

# import random
# document_length = 4000
# if len(corpus) > document_length:
#     start_index = random.randint(0, len(corpus) - document_length)
#     document = corpus[start_index : start_index + document_length]
# else:
#     document = corpus

# print(document)