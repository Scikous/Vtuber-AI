# from model_utils import LLMUtils
custom_model = "LLM/unnamedSICUACCT"

# from exllamav2 import ExLlamaV2
# from transformers import AutoTokenizer

# from exllamav2.generator import (
#     ExLlamaV2Sampler,
# )


# model_name = "TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ"
# #model, tokenizer = load_exllamav2_model(model_name)


# settings = ExLlamaV2Sampler.Settings()
# settings.temperature = 0.85
# settings.top_k = 50
# settings.top_p = 0.8
# settings.token_repetition_penalty = 1.05


# model = ExLlamaV2(model_path=model_name, settings=settings, streaming=True, max_new_tokens=150, verbose=True)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # LLMUtils.convert_model_to_Onnx(custom_model)



from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
import time
model_dir = "LLM/CapybaraHermes-2.5-Mistral-7B-GPTQ"
config = ExLlamaV2Config(model_dir)
model = ExLlamaV2(config)
cache = ExLlamaV2Cache(model, max_seq_len = 65536, lazy = True)
model.load_autosplit(cache, progress = True)
tokenizer = ExLlamaV2Tokenizer(config)

from exllamav2.generator import ExLlamaV2DynamicGenerator

generator = ExLlamaV2DynamicGenerator(
    model = model,
    cache = cache,
    tokenizer = tokenizer,
)

s = time.time()
satr = time.perf_counter()
output = generator.generate(
    prompt = ["Five good reasons to adopt a cat:","Tell 5 simple jokes:", "how much is 8 + 19?"],
    max_new_tokens = 200,
    add_bos = True)
end = time.perf_counter()
e = time.time()
print(output,satr,end, end-satr, e-s)


# def vllm_test():
#     from vllm import LLM, SamplingParams
#     prompts = [
#     "Hello, my name is",
#     "The president of the United States is",
#     "The capital of France is",
#     "The future of AI is",
#     ]
#     sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
#     llm = LLM(model=model_dir)
#     outputs = llm.generate(prompts, sampling_params)

# # Print the outputs.
#     for output in outputs:
#         prompt = output.prompt
#         generated_text = output.outputs[0].text
#         print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")