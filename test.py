from llama_index.core import VectorStoreIndex, PromptTemplate, ServiceContext


# chroma_collection = chroma_client.get_collection("ai_arxiv_full")
# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
# index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# Prompt template
with open("eval_questions/text_qa_template.txt", 'r', encoding='utf-8') as file:
    text_qa_template_str = file.read()

text_qa_template = PromptTemplate(text_qa_template_str)

print(text_qa_template)