INFO 03-26 14:46:20 [__init__.py:239] Automatically detected platform cuda.
/opt/conda/lib/python3.11/site-packages/pydub/utils.py:170: RuntimeWarning: Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work
  warn("Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work", RuntimeWarning)
INFO 03-26 14:46:22 [config.py:2610] Downcasting torch.float32 to torch.bfloat16.
INFO 03-26 14:46:27 [config.py:585] This model supports multiple tasks: {'reward', 'embed', 'score', 'generate', 'classify'}. Defaulting to 'generate'.
INFO 03-26 14:46:27 [config.py:1697] Chunked prefill is enabled with max_num_batched_tokens=2048.
WARNING 03-26 14:46:36 [utils.py:2181] We must use the `spawn` multiprocessing start method. Overriding VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. See https://docs.vllm.ai/en/latest/getting_started/troubleshooting.html#python-multiprocessing for more information. Reason: CUDA is initialized
INFO 03-26 14:46:40 [__init__.py:239] Automatically detected platform cuda.
/opt/conda/lib/python3.11/site-packages/pydub/utils.py:170: RuntimeWarning: Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work
  warn("Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work", RuntimeWarning)
INFO 03-26 14:46:42 [core.py:54] Initializing a V1 LLM engine (v0.8.2) with config: model='canopylabs/orpheus-3b-0.1-ft', speculative_config=None, tokenizer='canopylabs/orpheus-3b-0.1-ft', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=131072, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto,  device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='xgrammar', reasoning_backend=None), observability_config=ObservabilityConfig(show_hidden_metrics=False, otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=None, served_model_name=canopylabs/orpheus-3b-0.1-ft, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=True, chunked_prefill_enabled=True, use_async_output_proc=True, disable_mm_preprocessor_cache=False, mm_processor_kwargs=None, pooler_config=None, compilation_config={"level":3,"custom_ops":["none"],"splitting_ops":["vllm.unified_attention","vllm.unified_attention_with_output"],"use_inductor":true,"compile_sizes":[],"use_cudagraph":true,"cudagraph_num_of_warmups":1,"cudagraph_capture_sizes":[512,504,496,488,480,472,464,456,448,440,432,424,416,408,400,392,384,376,368,360,352,344,336,328,320,312,304,296,288,280,272,264,256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"max_capture_size":512}
WARNING 03-26 14:46:42 [utils.py:2321] Methods determine_num_available_blocks,device_config,get_cache_block_size_bytes,initialize_cache not implemented in <vllm.v1.worker.gpu_worker.Worker object at 0x70e690605290>
INFO 03-26 14:46:42 [parallel_state.py:954] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0
INFO 03-26 14:46:42 [cuda.py:220] Using Flash Attention backend on V1 engine.
INFO 03-26 14:46:43 [gpu_model_runner.py:1174] Starting to load model canopylabs/orpheus-3b-0.1-ft...
WARNING 03-26 14:46:43 [topk_topp_sampler.py:63] FlashInfer is not available. Falling back to the PyTorch-native implementation of top-p & top-k sampling. For the best performance, please install FlashInfer.
INFO 03-26 14:46:43 [weight_utils.py:265] Using model weights format ['*.safetensors']
INFO 03-26 14:46:44 [weight_utils.py:281] Time spent downloading weights for canopylabs/orpheus-3b-0.1-ft: 1.394894 seconds
Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:01,  2.02it/s]
Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:01<00:00,  2.47it/s]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.84it/s]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:02<00:00,  1.96it/s]

INFO 03-26 14:46:47 [loader.py:447] Loading weights took 2.18 seconds
INFO 03-26 14:46:47 [gpu_model_runner.py:1186] Model loading took 6.1801 GB and 4.645271 seconds
INFO 03-26 14:46:53 [backends.py:415] Using cache directory: /root/.cache/vllm/torch_compile_cache/94e9aa5bfb/rank_0_0 for vLLM's torch.compile
INFO 03-26 14:46:53 [backends.py:425] Dynamo bytecode transform time: 5.27 s
INFO 03-26 14:46:53 [backends.py:115] Directly load the compiled graph for shape None from the cache
INFO 03-26 14:46:58 [monitor.py:33] torch.compile takes 5.27 s in total
INFO 03-26 14:46:58 [kv_cache_utils.py:566] GPU KV cache size: 280,480 tokens
INFO 03-26 14:46:58 [kv_cache_utils.py:569] Maximum concurrency for 131,072 tokens per request: 2.14x
INFO 03-26 14:47:15 [gpu_model_runner.py:1534] Graph capturing finished in 17 secs, took 0.45 GiB
INFO 03-26 14:47:15 [core.py:151] init engine (profile, create kv cache, warmup model) took 27.91 seconds
ckpt path or config path does not exist! Downloading the model from the Hugging Face Hub...
Fetching 13 files: 100%|████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 549.19it/s]
Hello, let's see how well this thing works with a longer generation on my 12 gigabyte card.
INFO 03-26 14:47:24 [async_llm.py:221] Added request req-001.
It took 6.522598092007684 seconds to generate 5.38 seconds of audio
ckpt path or config path does not exist! Downloading the model from the Hugging Face Hub...
Fetching 13 files: 100%|█████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 125925.99it/s]
Watermark verification: Success
[rank0]:[W326 14:47:36.256942083 ProcessGroupNCCL.cpp:1496] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
root@failbowl:/app#
