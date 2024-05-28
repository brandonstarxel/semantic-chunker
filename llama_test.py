import transformers
import torch

# Check if MPS (Apple Silicon GPU) is available, otherwise use CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"

model_id = "meta-llama/Meta-Llama-3-8B"

pipeline = transformers.pipeline(
  "text-generation",
  model=model_id,
  model_kwargs={"torch_dtype": torch.bfloat16},
  device=device,
)
