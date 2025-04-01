# %%
import resource
import warnings
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import os
import sys
from typing import Any, Optional
import aiohttp
import numpy as np
import requests
from tqdm.asyncio import tqdm
import platform
import psutil
import subprocess
from datetime import datetime
import socket
import distro  # For detailed Linux distribution info
import cpuinfo  # For detailed CPU info


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


def remove_prefix(text: str, prefix: str) -> str:
    return text[len(prefix) :] if text.startswith(prefix) else text


def sample_gaussian_integer(min_val, max_val, mean=None, std=None):
    """
    Generate a random integer from a Gaussian-like distribution within specified bounds.
    
    Uses numpy's random.normal when available for efficiency, with fallback to a safer
    implementation of Box-Muller transform that handles edge cases better.
    
    Args:
        min_val: Minimum acceptable value (inclusive)
        max_val: Maximum acceptable value (inclusive)
        mean: Center of distribution (defaults to midpoint of range)
        std: Standard deviation (defaults to range/6 for ~99.7% coverage)
    
    Returns:
        An integer within the specified bounds
    """
    import time
    
    # Validate inputs to avoid issues
    min_val = int(min_val)
    max_val = int(max_val)
    
    if min_val > max_val:
        min_val, max_val = max_val, min_val
    
    if min_val == max_val:
        return min_val
    
    if mean is None:
        mean = (min_val + max_val) / 2
    if std is None:
        std = max(1, (max_val - min_val) / 6)  # Ensure std is at least 1
    
    # Set a maximum number of attempts to avoid infinite loops
    max_attempts = 100
    attempts = 0
    
    # Try the faster numpy approach first
    try:
        sample = int(round(np.random.normal(mean, std)))
        # If it's within bounds, return it
        if min_val <= sample <= max_val:
            return sample
    except Exception:
        # If numpy's normal fails, we'll use the fallback below
        pass
    
    # Fallback to a more robust custom implementation
    while attempts < max_attempts:
        attempts += 1
        try:
            # Box-Muller transform with additional safety checks
            u1 = np.random.random()
            if u1 < 1e-10:  # Avoid log(0)
                u1 = 1e-10
                
            u2 = np.random.random()
            
            # Generate normal random variable
            z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
            
            # Transform to desired mean and standard deviation
            sample = int(round(mean + z * std))
            
            # Only return if within bounds
            if min_val <= sample <= max_val:
                return sample
                
        except Exception as e:
            # If we hit any numerical errors, try again
            continue
    
    # If we've exceeded the maximum attempts, just return a uniform random integer
    return np.random.randint(min_val, max_val + 1)


# Todo: this should belong in the adapter classes
def get_model(host: str, port: str):
    model_url = f"http://{host}:{port}/v1/models"

    try:
        response = requests.get(model_url)
        model_list = response.json().get("data", [])
        model_id = model_list[0]["id"] if model_list else None
    except Exception as e:
        print(f"Failed to fetch model from {model_url}. Error: {e}")
        print("Please specify the correct host and port using `--host` and `--port`.")
        sys.exit(1)
    return model_id


def get_tokenizer(model: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model,
        trust_remote_code=False,
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        warnings.warn(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead."
        )
    return tokenizer


def is_notebook():
    return "ipykernel" in sys.modules


def download_and_cache_file(url: str, filename: Optional[str] = None):
    """Read and cache a file from a url."""
    import time
    start_time = time.time()
    
    if filename is None:
        filename = os.path.join("/tmp", url.split("/")[-1])

    # Check if the cache file already exists
    if os.path.exists(filename):
        file_size = os.path.getsize(filename)
        print(f"Using cached file at {filename} ({file_size / (1024*1024):.2f} MB)")
        return filename

    print(f"[{time.time() - start_time:.2f}s] Downloading from {url} to {filename}")

    try:
        # Stream the response to show the progress bar
        response = requests.get(url, stream=True, timeout=60)  # Add timeout
        response.raise_for_status()  # Check for request errors

        # Total size of the file in bytes
        total_size = int(response.headers.get("content-length", 0))
        print(f"[{time.time() - start_time:.2f}s] File size: {total_size / (1024*1024):.2f} MB")
        chunk_size = 8192  # Increased chunk size for better performance (8KB)

        # Use tqdm to display the progress bar
        with open(filename, "wb") as f, tqdm(
            desc=filename,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            download_start = time.time()
            bytes_downloaded = 0
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                bytes_downloaded += len(chunk)
                bar.update(len(chunk))
                
                # Log progress every 10MB
                if bytes_downloaded % (10 * 1024 * 1024) < chunk_size:
                    elapsed = time.time() - download_start
                    speed = bytes_downloaded / (1024 * 1024 * elapsed) if elapsed > 0 else 0
                    print(f"[{time.time() - start_time:.2f}s] Downloaded {bytes_downloaded / (1024*1024):.2f} MB at {speed:.2f} MB/s")
            
        print(f"[{time.time() - start_time:.2f}s] Download completed")
        return filename
    except Exception as e:
        print(f"[{time.time() - start_time:.2f}s] Download failed: {str(e)}")
        # If the file was partially created, remove it
        if os.path.exists(filename):
            os.remove(filename)
        raise


def set_fd_limit(target=65535):
    resource_type = resource.RLIMIT_NOFILE
    soft, hard = resource.getrlimit(resource_type)

    if soft < target:
        try:
            resource.setrlimit(resource_type, (target, hard))
        except ValueError as e:
            print(f"Fail to set RLIMIT_NOFILE: {e}")


def get_gpu_info():
    """Gather GPU information using nvidia-smi if available"""
    try:
        nvidia_smi = "nvidia-smi"
        result = subprocess.check_output(
            [
                nvidia_smi,
                "--query-gpu=gpu_name,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
        gpu_info = []
        for line in result.strip().split("\n"):
            name, total, free, used, temp, util = line.split(", ")
            gpu_info.append(
                {
                    "name": name,
                    "memory_total_mb": float(total),
                    "memory_free_mb": float(free),
                    "memory_used_mb": float(used),
                    "temperature_c": float(temp),
                    "utilization_percent": float(util),
                }
            )
        return gpu_info
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def get_torch_info():
    """Get PyTorch-specific information if available"""

    try:
        import torch

        return {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,  # type: ignore
            "cudnn_version": torch.backends.cudnn.version()
            if torch.cuda.is_available()
            else None,
            "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_names": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else [],
        }
    except ImportError:
        return None


def get_system_info():
    """Gather comprehensive system information"""
    cpu = cpuinfo.get_cpu_info()
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    # Helper function to convert bytes to GiB
    def bytes_to_gib(bytes_value):
        return bytes_value / (1024**3)

    system_info = {
        "timestamp": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        # OS Information
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "linux_distribution": distro.info()
            if platform.system() == "Linux"
            else None,
        },
        # CPU Information
        "cpu": {
            "brand": cpu.get("brand_raw", "Unknown"),
            "architecture": cpu.get("arch", "Unknown"),
            "bits": cpu.get("bits", "Unknown"),
            "count_physical": psutil.cpu_count(logical=False),
            "count_logical": psutil.cpu_count(logical=True),
            "frequency_mhz": {
                "current": psutil.cpu_freq().current if psutil.cpu_freq() else None,
                "min": psutil.cpu_freq().min if psutil.cpu_freq() else None,
                "max": psutil.cpu_freq().max if psutil.cpu_freq() else None,
            },
            "load_percent": psutil.cpu_percent(interval=1, percpu=True),
        },
        # Memory Information
        "memory": {
            "total_gib": bytes_to_gib(memory.total),
            "available_gib": bytes_to_gib(memory.available),
            "used_gib": bytes_to_gib(memory.used),
            "free_gib": bytes_to_gib(memory.free),
            "percent_used": memory.percent,
            "swap": {
                "total_gib": bytes_to_gib(psutil.swap_memory().total),
                "used_gib": bytes_to_gib(psutil.swap_memory().used),
                "free_gib": bytes_to_gib(psutil.swap_memory().free),
                "percent_used": psutil.swap_memory().percent,
            },
        },
        # Disk Information
        "disk": {
            "total_gib": bytes_to_gib(disk.total),
            "used_gib": bytes_to_gib(disk.used),
            "free_gib": bytes_to_gib(disk.free),
            "percent_used": disk.percent,
        },
        # Network Information
        "network": {
            "interfaces": {name: addr for name, addr in psutil.net_if_addrs().items()},
            "connections": len(psutil.net_connections()),
        },
        # Python Information
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
            "executable": sys.executable,
        },
        # Environment Variables (filtered)
        "env": {
            k: v
            for k, v in os.environ.items()
            if any(
                key in k.lower()
                for key in ["cuda", "python", "path", "library", "ld_library"]
            )
        },
    }

    # Add GPU information if available
    gpu_info = get_gpu_info()
    if gpu_info:
        system_info["gpu"] = gpu_info

    # Add PyTorch information if available
    torch_info = get_torch_info()
    if torch_info:
        system_info["pytorch"] = torch_info

    return system_info
