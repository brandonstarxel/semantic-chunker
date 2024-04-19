From AI21 Labs:
https://python.langchain.com/docs/integrations/document_transformers/ai21_semantic_text_splitter/

Get AI21 API Key?

From Greg Kamradt:
https://python.langchain.com/docs/modules/data_connection/document_transformers/semantic-chunker/


Project notes:

Retrieval Precision captures if we're retrieved the correct text but
does not capture if our context window is too large.
Answer similirity would measure that . . .



First results set with arXiv paper dataset, 107 questions:
chunk_1:
RecursiveCharacterTextSplitter
chunk_size=400
chunk_overlap=200
embedding-3-small
Precision at 1: 0.9252336448598131
Precision at 2: 0.9065420560747663
Precision at 3: 0.9034267912772584
Precision at 4: 0.8925233644859814
Precision at 5: 0.8766355140186917

chunk_2:
TokenTextSplitter
chunk_size=400
chunk_overlap=200
embedding-3-small
Precision at 1: 0.9065420560747663
Precision at 2: 0.897196261682243
Precision at 3: 0.8878504672897196
Precision at 4: 0.8738317757009346
Precision at 5: 0.8598130841121495

chunk_3:
TokenTextSplitter
chunk_size=400
chunk_overlap=0
embedding-3-small
Precision at 1: 0.8878504672897196
Precision at 2: 0.8457943925233645
Precision at 3: 0.8348909657320872
Precision at 4: 0.8154205607476636
Precision at 5: 0.8037383177570093