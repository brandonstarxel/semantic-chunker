# from chromadb.utils.embedding_functions
from tokenizers import Tokenizer 
import os
from pathlib import Path
from typing import cast


MODEL_NAME = "all-MiniLM-L6-v2"
DOWNLOAD_PATH = Path.home() / ".cache" / "chroma" / "onnx_models" / MODEL_NAME
EXTRACTED_FOLDER_NAME = "onnx"
ARCHIVE_FILENAME = "onnx.tar.gz"
MODEL_DOWNLOAD_URL = (
    "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"
)
_MODEL_SHA256 = "913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3"


tokenizer = Tokenizer.from_file(
    os.path.join(
        DOWNLOAD_PATH, EXTRACTED_FOLDER_NAME, "tokenizer.json"
    )
)

tokenizer = cast(Tokenizer, tokenizer)

tokenizer.enable_truncation(max_length=256)

tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=256)

tokens = tokenizer.encode("Hello, world!").tokens

# tokens = tokenizer.encode("Hello, world!"*100).tokens

print([token for token in tokens if token != "[PAD]"])
print(len([token for token in tokens if token != "[PAD]"]))