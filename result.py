from datetime import datetime
import json
from typing import Any, List
import warnings
from dataclasses import dataclass, asdict
import numpy as np
from transformers import (
    PreTrainedTokenizerBase,
)
from model import BenchmarkConfig
from utils import get_system_info
from simulation import SimulationResult


@dataclass
class BenchmarkResult:
    completed: int
    total_time: float
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    mean_ttft_ms: float
    p50_ttft_ms: float  # renamed from median
    p75_ttft_ms: float  # new
    p95_ttft_ms: float  # new
    p99_ttft_ms: float
    std_ttft_ms: float
    mean_tpot_ms: float
    p50_tpot_ms: float  # renamed from median
    p75_tpot_ms: float  # new
    p95_tpot_ms: float  # new
    p99_tpot_ms: float
    std_tpot_ms: float
    mean_itl_ms: float
    p50_itl_ms: float  # renamed from median
    p75_itl_ms: float  # new
    p95_itl_ms: float  # new
    p99_itl_ms: float
    std_itl_ms: float
    mean_e2e_latency_ms: float
    p50_e2e_latency_ms: float  # renamed from median
    errors: List[Any]


def process_result(
    result: SimulationResult,
    tokenizer: PreTrainedTokenizerBase,
) -> BenchmarkResult:
    output_lens: List[int] = []
    total_input = 0
    completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    e2e_latencies: List[float] = []
    errors = []

    turn_results = result.turn_results
    duration = result.duration

    for i, res in enumerate(turn_results):
        if res.success:
            retokenized_len = len(tokenizer.encode(res.completion))
            output_lens.append(retokenized_len)
            if retokenized_len > 1:
                tpots.append((res.latency - res.ttft) / (retokenized_len - 1))
            itls += res.itl
            ttfts.append(res.ttft)
            total_input += res.prompt_len
            e2e_latencies.append(res.latency)
            completed += 1
        else:
            output_lens.append(0)
            errors.append(res.error)

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )
    return BenchmarkResult(
        completed=completed,
        total_time=duration,
        total_input=total_input,
        total_output=sum(output_lens),
        request_throughput=completed / duration,
        input_throughput=total_input / duration,
        output_throughput=sum(output_lens) / duration,
        mean_ttft_ms=float(np.mean(ttfts or [0])) * 1000,
        p50_ttft_ms=float(np.median(ttfts or [0])) * 1000,
        p75_ttft_ms=float(np.percentile(ttfts or [0], 75)) * 1000,
        p95_ttft_ms=float(np.percentile(ttfts or [0], 95)) * 1000,
        p99_ttft_ms=float(np.percentile(ttfts or [0], 99)) * 1000,
        std_ttft_ms=float(np.std(ttfts or [0])) * 1000,
        mean_tpot_ms=float(np.mean(tpots or [0])) * 1000,
        p50_tpot_ms=float(np.median(tpots or [0])) * 1000,
        p75_tpot_ms=float(np.percentile(tpots or [0], 75)) * 1000,
        p95_tpot_ms=float(np.percentile(tpots or [0], 95)) * 1000,
        p99_tpot_ms=float(np.percentile(tpots or [0], 99)) * 1000,
        std_tpot_ms=float(np.std(tpots or [0])) * 1000,
        mean_itl_ms=float(np.mean(itls or [0])) * 1000,
        p50_itl_ms=float(np.median(itls or [0])) * 1000,
        p75_itl_ms=float(np.percentile(itls or [0], 75)) * 1000,
        p95_itl_ms=float(np.percentile(itls or [0], 95)) * 1000,
        p99_itl_ms=float(np.percentile(itls or [0], 99)) * 1000,
        std_itl_ms=float(np.std(itls or [0])) * 1000,
        mean_e2e_latency_ms=float(np.mean(e2e_latencies)) * 1000,
        p50_e2e_latency_ms=float(np.median(e2e_latencies)) * 1000,
        errors=errors,
    )


def show_result(result: BenchmarkResult, config: BenchmarkConfig):
    print("\n{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Backend:", config.backend))
    print("{:<40} {:<10}".format("Successful requests:", result.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", result.total_time))
    print("{:<40} {:<10}".format("Total input tokens:", result.total_input))
    print(
        "{:<40} {:<10}".format(
            "Total generated tokens (retokenized):", result.total_output
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", result.request_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Input token throughput (tok/s):", result.input_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (retokenized) (tok/s):",
            result.output_throughput,
        )
    )
    print("{s:{c}^{n}}".format(s="End-to-End Latency", n=50, c="-"))
    print(
        "{:<40} {:<10.2f}".format("Mean E2E Latency (ms):", result.mean_e2e_latency_ms)
    )
    print("{:<40} {:<10.2f}".format("P50 E2E Latency (ms):", result.p50_e2e_latency_ms))
    print("{s:{c}^{n}}".format(s="Time to First Token", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", result.mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("P50 TTFT (ms):", result.p50_ttft_ms))
    print("{:<40} {:<10.2f}".format("P75 TTFT (ms):", result.p75_ttft_ms))
    print("{:<40} {:<10.2f}".format("P95 TTFT (ms):", result.p95_ttft_ms))
    print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", result.p99_ttft_ms))
    print(
        "{s:{c}^{n}}".format(s="Time per Output Token (excl. 1st token)", n=50, c="-")
    )
    print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", result.mean_tpot_ms))
    print("{:<40} {:<10.2f}".format("P50 TPOT (ms):", result.p50_tpot_ms))
    print("{:<40} {:<10.2f}".format("P75 TPOT (ms):", result.p75_tpot_ms))
    print("{:<40} {:<10.2f}".format("P95 TPOT (ms):", result.p95_tpot_ms))
    print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", result.p99_tpot_ms))
    print("{s:{c}^{n}}".format(s="Inter-token Latency", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean ITL (ms):", result.mean_itl_ms))
    print("{:<40} {:<10.2f}".format("P50 ITL (ms):", result.p50_itl_ms))
    print("{:<40} {:<10.2f}".format("P75 ITL (ms):", result.p75_itl_ms))
    print("{:<40} {:<10.2f}".format("P95 ITL (ms):", result.p95_itl_ms))
    print("{:<40} {:<10.2f}".format("P99 ITL (ms):", result.p99_itl_ms))
    print("=" * 50)


def persist_result(result: BenchmarkResult, config: BenchmarkConfig):
    iso = datetime.now().strftime("%Y%m%d%H%M%S")
    obj = asdict(result)
    obj["args"] = vars(config)
    obj["system"] = get_system_info()

    with open(f"{iso}-hyperbench.json", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)
