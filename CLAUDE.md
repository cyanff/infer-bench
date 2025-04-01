# infer-bench Reference Guide

## Run Commands
- Install: `pip install -r requirements.txt`
- Run benchmark: `python bench.py --model <model> --backend <backend> --ccu <concurrent_users> --time <duration>`
- Common backends: `sglang`, `lmdeploy`, `vllm`

## Code Style
- Imports: stdlib first, then third-party, then local imports
- Formatting: Use black with default settings
- Types: All functions should have type annotations (input and return)
- Naming: snake_case for variables/functions, PascalCase for classes
- Error handling: Catch specific exceptions with detailed error messages
- Dataclasses preferred for structured data
- Use docstrings for functions and classes (see `sample_gaussian_integer`)
- Performance-critical code should be async whenever possible
- Comments explain "why" not "what" (implementation details)