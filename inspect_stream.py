from mlx_lm import stream_generate, load
from mlx_lm.sample_utils import make_sampler
import sys

# Mock model loading to avoid heavy load if possible, or just load the model if it's already there.
# Since loading 70B takes time, maybe I can just inspect the function signature/docstring for GenerationResponse?
# Or try to import GenerationResponse.

try:
    from mlx_lm.utils import GenerationResponse
    print("Found GenerationResponse in mlx_lm.utils")
    print(dir(GenerationResponse))
except ImportError:
    print("GenerationResponse not found in mlx_lm.utils")

# Let's try to inspect stream_generate return annotation
import inspect
sig = inspect.signature(stream_generate)
print(f"stream_generate signature: {sig}")
