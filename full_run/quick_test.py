from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from utils import count_non_pad_tokens, num_tokens_from_string

# chunker = SentenceTransformersTokenTextSplitter(
#             # chunk_size=200,
#             tokens_per_chunk=100,
#             chunk_overlap=0,
#             model_name="sentence-transformers/all-MiniLM-L6-v2",
#             add_start_index=True
#         )

# from langchain_core.documents import Document

# document = Document(page_content="This is a test sentence. HELLO!"*100)
# results = chunker.create_documents(["This is a test sentence. HELLO!"*100])

# print(results)

test_str = """
 = Valkyria Chronicles III = 
 Senjō no Valkyria 3 : <unk> Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria series . Employing the same fusion of tactical and real @-@ time gameplay as its predecessors , the story runs parallel to the first game and follows the " Nameless " , a penal military unit serving the nation of Gallia during the Second Europan War who perform secret black operations and are pitted against the Imperial unit " <unk> Raven " . 
 The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . While it retained the standard features of the series , it also underwent multiple adjustments , such as making the game more forgiving for series newcomers . Character designer <unk> Honjou and composer Hitoshi Sakimoto both returned from previous entries , along with Valkyria Chronicles II director Takeshi Ozawa . A large team of writers handled the script . The game 's opening theme was sung by May 'n . 
 It met with positive sales in Japan , and
"""

print(count_non_pad_tokens(test_str))
print(num_tokens_from_string(test_str))