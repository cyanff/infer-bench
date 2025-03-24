import asyncio
import argparse
from cache import Cache
from utils import get_model
from result import process_result, show_result, persist_result
from simulation import Simulation
from transformers import AutoTokenizer
from model import BenchmarkConfig


"""
TODO:
Make this script manage the benchmark end to end.
It should:
- Spin up the inference server
- Start the benchmark
- Restart the inference server 
- Recurse and grid search over params
"""


backend_config = {
    "sglang": {"host": "127.0.0.1", "port": "30000"},
    "lmdeploy": {"host": "127.0.0.1", "port": "8080"},
    "vllm": {"host": "127.0.0.1", "port": "8000"},
}


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(
        description="Run benchmarking simulation with configurable parameters"
    )
    parser.add_argument(
        "--model", type=str, help="Model to use for simulation (optional)", default=None
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        help="Model name to request from the inference API (defaults to --model if not specified)",
        default=None,
    )
    parser.add_argument(
        "--backend",
        type=str,
        help="The backend to benchmark, i.e 'sglang', 'lmdeploy', 'vllm'.",
    )
    parser.add_argument("--host", type=str, help="Ex: 127.0.0.1")
    parser.add_argument("--port", type=str, help="Ex: 8080")
    parser.add_argument("--max-prompt", type=int, default=2048)
    parser.add_argument("--min-completion", type=int, default=100)
    parser.add_argument("--max-completion", type=int, default=250)
    parser.add_argument("--min-cd", type=int, default=7)
    parser.add_argument("--max-cd", type=int, default=14)

    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Tokenizer to use",
    )
    parser.add_argument(
        "--ccu", type=int, help="Concurrent users (optional)", default=250
    )
    parser.add_argument("--time", type=int, help="Time limit in seconds", default=30)
    args = parser.parse_args()

    if (
        args.model is None
        and (args.host is None or args.port is None)
        and args.backend is None
    ):
        raise Exception(
            "You must specify a model, or provide host and port, or specify a backend."
        )

    if bool(args.host) ^ bool(args.port):
        raise Exception("Both host and port must be specified")

    if args.host is None and args.backend is None:
        raise Exception(
            "You must either provide host and port or specify a backend through --backend"
        )
    host = args.host
    port = args.port
    if host is None:
        host = backend_config[args.backend]["host"]
        port = backend_config[args.backend]["port"]

    model = args.model
    if model is None:
        # Use provided model or get default
        model = args.model if args.model else get_model(host, port)
        if model is None:
            raise Exception(
                "No model was specified, failed to get default model from endpoint"
            )
    return BenchmarkConfig(
        model=model,
        tokenizer=args.tokenizer or model,
        backend=args.backend,
        ccu=args.ccu,
        host=host,
        port=port,
        max_prompt=args.max_prompt,
        min_completion=args.min_completion,
        max_completion=args.max_completion,
        min_cd=args.min_cd,
        max_cd=args.max_cd,
        time=args.time,
        served_model_name=args.served_model_name,
    )


async def main():
    cfg = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
    sim = Simulation(
        model=cfg.model,
        tokenizer=tokenizer,
        backend_kind=cfg.backend,
        tokenizer_id=cfg.tokenizer,
        cache=Cache(),
        ccu=cfg.ccu,
        timelimit=cfg.time,
        min_cd=cfg.min_cd,
        max_cd=cfg.max_cd,
        min_completion=cfg.min_completion,
        max_completion=cfg.max_completion,
        served_model_name=cfg.served_model_name,
    )
    sim_res = await sim.boot()
    res = process_result(result=sim_res, tokenizer=tokenizer)
    show_result(res, cfg)
    persist_result(res, cfg)


if __name__ == "__main__":
    asyncio.run(main())
