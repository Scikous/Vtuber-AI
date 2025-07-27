import os
import torch
import multiprocessing as mp
import gc
import asyncio

def apply_system_optimizations(logger, use_cuda=True, num_threads=None):
    """Apply system-level optimizations for minimal latency."""
    logger.info("🔧 Applying system-level optimizations...")
    
    if use_cuda and torch.cuda.is_available():
        logger.info("⚡ Applying CUDA optimizations...")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    logger.info("🚀 Applying CPU optimizations...")
    num_threads = num_threads or max(1, mp.cpu_count() // 2)
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(max(1, num_threads // 2))
    
    gc.collect()
    
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)
    
    logger.info("✅ System optimizations applied!")

async def check_gpu_memory(logger):
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        if total > 0:
            while True:
                allocated = torch.cuda.memory_allocated()
                if allocated / total <= 0.9:
                    break
                logger.warning("GPU memory usage high, waiting...")
                await asyncio.sleep(1)

def get_cuda_utilization():
    if not torch.cuda.is_available():
        return None # Or raise an error, or return 0.0 for no GPU

    allocated_mem = torch.cuda.memory_allocated()
    max_allocated_mem = torch.cuda.max_memory_allocated()

    if max_allocated_mem > 0:
        return allocated_mem / max_allocated_mem
    else:
        # If max_allocated_mem is 0, it means no memory has been used yet,
        # so utilization is effectively 0%.
        return 0.0