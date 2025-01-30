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
    if mean is None:
        mean = (min_val + max_val) / 2
    if std is None:
        std = (max_val - min_val) / 6

    while True:
        sample = int(round(np.random.normal(mean, std)))
        if min_val <= sample <= max_val:
            return sample


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
    if filename is None:
        filename = os.path.join("/tmp", url.split("/")[-1])

    # Check if the cache file already exists
    if os.path.exists(filename):
        return filename

    print(f"Downloading from {url} to {filename}")

    # Stream the response to show the progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check for request errors

    # Total size of the file in bytes
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024  # Download in chunks of 1KB

    # Use tqdm to display the progress bar
    with open(filename, "wb") as f, tqdm(
        desc=filename,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    return filename


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
