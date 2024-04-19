import tiktoken

# Count the number of tokens in each page_content
def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string, disallowed_special=()))
    return num_tokens

def get_retrieval_precision_prompt(question, context):
    main_message = ("Considering the following question and context, determine whether the context "
                    "is relevant for answering the question. If the context is relevant for "
                    "answering the question, respond with true. If the context is not relevant for "
                    "answering the question, respond with false. Respond with either true or false "
                    "and no additional text.")

    main_message += f"\nQUESTION: {question}\n"
    main_message += f"CONTEXT: {context}\n"

    return main_message