INFO 04-06 16:17:38 [__init__.py:239] Automatically detected platform cuda.
usage: vllm serve [model_tag] [options]

positional arguments:
  model_tag             The model tag to serve (optional if specified in
                        config) (default: None)

options:
  --additional-config ADDITIONAL_CONFIG
                        Additional config for specified platform in JSON
                        format. Different platforms may support different
                        configs. Make sure the configs are valid for the
                        platform you are using. The input format is like
                        '{"config_key":"config_value"}' (default: None)
  --allow-credentials   Allow credentials. (default: False)
  --allowed-headers ALLOWED_HEADERS
                        Allowed headers. (default: ['*'])
  --allowed-local-media-path ALLOWED_LOCAL_MEDIA_PATH
                        Allowing API requests to read local images or videos
                        from directories specified by the server file system.
                        This is a security risk. Should only be enabled in
                        trusted environments. (default: None)
  --allowed-methods ALLOWED_METHODS
                        Allowed methods. (default: ['*'])
  --allowed-origins ALLOWED_ORIGINS
                        Allowed origins. (default: ['*'])
  --api-key API_KEY     If provided, the server will require this key to be
                        presented in the header. (default: None)
  --block-size {8,16,32,64,128}
                        Token block size for contiguous chunks of tokens. This
                        is ignored on neuron devices and set to ``--max-model-
                        len``. On CUDA devices, only block sizes up to 32 are
                        supported. On HPU devices, block size defaults to 128.
                        (default: None)
  --calculate-kv-scales
                        This enables dynamic calculation of k_scale and
                        v_scale when kv-cache-dtype is fp8. If calculate-kv-
                        scales is false, the scales will be loaded from the
                        model checkpoint if available. Otherwise, the scales
                        will default to 1.0. (default: False)
  --chat-template CHAT_TEMPLATE
                        The file path to the chat template, or the template in
                        single-line form for the specified model. (default:
                        None)
  --chat-template-content-format {auto,string,openai}
                        The format to render message content within a chat
                        template. * "string" will render the content as a
                        string. Example: ``"Hello World"`` * "openai" will
                        render the content as a list of dictionaries, similar
                        to OpenAI schema. Example: ``[{"type": "text", "text":
                        "Hello world!"}]`` (default: auto)
  --code-revision CODE_REVISION
                        The specific revision to use for the model code on
                        Hugging Face Hub. It can be a branch name, a tag name,
                        or a commit id. If unspecified, will use the default
                        version. (default: None)
  --collect-detailed-traces COLLECT_DETAILED_TRACES
                        Valid choices are model,worker,all. It makes sense to
                        set this only if ``--otlp-traces-endpoint`` is set. If
                        set, it will collect detailed traces for the specified
                        modules. This involves use of possibly costly and or
                        blocking operations and hence might have a performance
                        impact. (default: None)
  --compilation-config COMPILATION_CONFIG, -O COMPILATION_CONFIG
                        torch.compile configuration for the model.When it is a
                        number (0, 1, 2, 3), it will be interpreted as the
                        optimization level. NOTE: level 0 is the default level
                        without any optimization. level 1 and 2 are for
                        internal testing only. level 3 is the recommended
                        level for production. To specify the full compilation
                        config, use a JSON string. Following the convention of
                        traditional compilers, using -O without space is also
                        supported. -O3 is equivalent to -O 3. (default: None)
  --config CONFIG       Read CLI options from a config file.Must be a YAML
                        with the following options:https://docs.vllm.ai/en/lat
                        est/serving/openai_compatible_server.html#cli-
                        reference (default: )
  --config-format {auto,hf,mistral}
                        The format of the model config to load. * "auto" will
                        try to load the config in hf format if available else
                        it will try to load in mistral format (default:
                        ConfigFormat.AUTO)
  --cpu-offload-gb CPU_OFFLOAD_GB
                        The space in GiB to offload to CPU, per GPU. Default
                        is 0, which means no offloading. Intuitively, this
                        argument can be seen as a virtual way to increase the
                        GPU memory size. For example, if you have one 24 GB
                        GPU and set this to 10, virtually you can think of it
                        as a 34 GB GPU. Then you can load a 13B model with
                        BF16 weight, which requires at least 26GB GPU memory.
                        Note that this requires fast CPU-GPU interconnect, as
                        part of the model is loaded from CPU memory to GPU
                        memory on the fly in each model forward pass.
                        (default: 0)
  --data-parallel-size DATA_PARALLEL_SIZE, -dp DATA_PARALLEL_SIZE
                        Number of data parallel replicas. MoE layers will be
                        sharded according to the product of the tensor-
                        parallel-size and data-parallel-size. (default: 1)
  --device {auto,cuda,neuron,cpu,tpu,xpu,hpu}
                        Device type for vLLM execution. (default: auto)
  --disable-async-output-proc
                        Disable async output processing. This may result in
                        lower performance. (default: False)
  --disable-cascade-attn
                        Disable cascade attention for V1. While cascade
                        attention does not change the mathematical
                        correctness, disabling it could be useful for
                        preventing potential numerical issues. Note that even
                        if this is set to False, cascade attention will be
                        only used when the heuristic tells that it's
                        beneficial. (default: False)
  --disable-custom-all-reduce
                        See ParallelConfig. (default: False)
  --disable-fastapi-docs
                        Disable FastAPI's OpenAPI schema, Swagger UI, and
                        ReDoc endpoint. (default: False)
  --disable-frontend-multiprocessing
                        If specified, will run the OpenAI frontend server in
                        the same process as the model serving engine.
                        (default: False)
  --disable-log-requests
                        Disable logging requests. (default: False)
  --disable-log-stats   Disable logging statistics. (default: False)
  --disable-mm-preprocessor-cache
                        If true, then disables caching of the multi-modal
                        preprocessor/mapper. (not recommended) (default:
                        False)
  --disable-sliding-window
                        Disables sliding window, capping to sliding window
                        size. (default: False)
  --disable-uvicorn-access-log
                        Disable uvicorn access log. (default: False)
  --distributed-executor-backend {ray,mp,uni,external_launcher}
                        Backend to use for distributed model workers, either
                        "ray" or "mp" (multiprocessing). If the product of
                        pipeline_parallel_size and tensor_parallel_size is
                        less than or equal to the number of GPUs available,
                        "mp" will be used to keep processing on a single host.
                        Otherwise, this will default to "ray" if Ray is
                        installed and fail otherwise. Note that tpu only
                        supports Ray for distributed inference. (default:
                        None)
  --download-dir DOWNLOAD_DIR
                        Directory to download and load the weights. (default:
                        None)
  --dtype {auto,half,float16,bfloat16,float,float32}
                        Data type for model weights and activations. * "auto"
                        will use FP16 precision for FP32 and FP16 models, and
                        BF16 precision for BF16 models. * "half" for FP16.
                        Recommended for AWQ quantization. * "float16" is the
                        same as "half". * "bfloat16" for a balance between
                        precision and range. * "float" is shorthand for FP32
                        precision. * "float32" for FP32 precision. (default:
                        auto)
  --enable-auto-tool-choice
                        Enable auto tool choice for supported models. Use
                        ``--tool-call-parser`` to specify which parser to use.
                        (default: False)
  --enable-chunked-prefill [ENABLE_CHUNKED_PREFILL]
                        If set, the prefill requests can be chunked based on
                        the max_num_batched_tokens. (default: None)
  --enable-expert-parallel
                        Use expert parallelism instead of tensor parallelism
                        for MoE layers. (default: False)
  --enable-lora         If True, enable handling of LoRA adapters. (default:
                        False)
  --enable-lora-bias    If True, enable bias for LoRA adapters. (default:
                        False)
  --enable-prefix-caching, --no-enable-prefix-caching
                        Enables automatic prefix caching. Use ``--no-enable-
                        prefix-caching`` to disable explicitly. (default:
                        None)
  --enable-prompt-adapter
                        If True, enable handling of PromptAdapters. (default:
                        False)
  --enable-prompt-tokens-details
                        If set to True, enable prompt_tokens_details in usage.
                        (default: False)
  --enable-reasoning    Whether to enable reasoning_content for the model. If
                        enabled, the model will be able to generate reasoning
                        content. (default: False)
  --enable-request-id-headers
                        If specified, API server will add X-Request-Id header
                        to responses. Caution: this hurts performance at high
                        QPS. (default: False)
  --enable-server-load-tracking
                        If set to True, enable tracking server_load_metrics in
                        the app state. (default: False)
  --enable-sleep-mode   Enable sleep mode for the engine. (only cuda platform
                        is supported) (default: False)
  --enable-ssl-refresh  Refresh SSL Context when SSL certificate files change
                        (default: False)
  --enforce-eager       Always use eager-mode PyTorch. If False, will use
                        eager mode and CUDA graph in hybrid for maximal
                        performance and flexibility. (default: False)
  --fully-sharded-loras
                        By default, only half of the LoRA computation is
                        sharded with tensor parallelism. Enabling this will
                        use the fully sharded layers. At high sequence length,
                        max rank or tensor parallel size, this is likely
                        faster. (default: False)
  --generation-config GENERATION_CONFIG
                        The folder path to the generation config. Defaults to
                        'auto', the generation config will be loaded from
                        model path. If set to 'vllm', no generation config is
                        loaded, vLLM defaults will be used. If set to a folder
                        path, the generation config will be loaded from the
                        specified folder path. If `max_new_tokens` is
                        specified in generation config, then it sets a server-
                        wide limit on the number of output tokens for all
                        requests. (default: auto)
  --gpu-memory-utilization GPU_MEMORY_UTILIZATION
                        The fraction of GPU memory to be used for the model
                        executor, which can range from 0 to 1. For example, a
                        value of 0.5 would imply 50% GPU memory utilization.
                        If unspecified, will use the default value of 0.9.
                        This is a per-instance limit, and only applies to the
                        current vLLM instance.It does not matter if you have
                        another vLLM instance running on the same GPU. For
                        example, if you have two vLLM instances running on the
                        same GPU, you can set the GPU memory utilization to
                        0.5 for each instance. (default: 0.9)
  --guided-decoding-backend GUIDED_DECODING_BACKEND
                        Which engine will be used for guided decoding (JSON
                        schema / regex etc) by default. Currently support
                        https://github.com/mlc-ai/xgrammar and
                        https://github.com/guidance-ai/llguidance.Valid
                        backend values are "xgrammar", "guidance", and "auto".
                        With "auto", we will make opinionated choices based on
                        requestcontents and what the backend libraries
                        currently support, so the behavior is subject to
                        change in each release. (default: xgrammar)
  --hf-config-path HF_CONFIG_PATH
                        Name or path of the huggingface config to use. If
                        unspecified, model name or path will be used.
                        (default: None)
  --hf-overrides HF_OVERRIDES
                        Extra arguments for the HuggingFace config. This
                        should be a JSON string that will be parsed into a
                        dictionary. (default: None)
  --host HOST           Host name. (default: None)
  --ignore-patterns IGNORE_PATTERNS
                        The pattern(s) to ignore when loading the
                        model.Default to `original/**/*` to avoid repeated
                        loading of llama's checkpoints. (default: [])
  --kv-cache-dtype {auto,fp8,fp8_e5m2,fp8_e4m3}
                        Data type for kv cache storage. If "auto", will use
                        model data type. CUDA 11.8+ supports fp8 (=fp8_e4m3)
                        and fp8_e5m2. ROCm (AMD GPU) supports fp8 (=fp8_e4m3)
                        (default: auto)
  --kv-transfer-config KV_TRANSFER_CONFIG
                        The configurations for distributed KV cache transfer.
                        Should be a JSON string. (default: None)
  --limit-mm-per-prompt LIMIT_MM_PER_PROMPT
                        For each multimodal plugin, limit how many input
                        instances to allow for each prompt. Expects a comma-
                        separated list of items, e.g.: `image=16,video=2`
                        allows a maximum of 16 images and 2 videos per prompt.
                        Defaults to 1 for each modality. (default: None)
  --load-format {auto,pt,safetensors,npcache,dummy,tensorizer,sharded_state,gguf,bitsandbytes,mistral,runai_streamer,fastsafetensors}
                        The format of the model weights to load. * "auto" will
                        try to load the weights in the safetensors format and
                        fall back to the pytorch bin format if safetensors
                        format is not available. * "pt" will load the weights
                        in the pytorch bin format. * "safetensors" will load
                        the weights in the safetensors format. * "npcache"
                        will load the weights in pytorch format and store a
                        numpy cache to speed up the loading. * "dummy" will
                        initialize the weights with random values, which is
                        mainly for profiling. * "tensorizer" will load the
                        weights using tensorizer from CoreWeave. See the
                        Tensorize vLLM Model script in the Examples section
                        for more information. * "runai_streamer" will load the
                        Safetensors weights using Run:aiModel Streamer. *
                        "bitsandbytes" will load the weights using
                        bitsandbytes quantization. * "sharded_state" will load
                        weights from pre-sharded checkpoint files, supporting
                        efficient loading of tensor-parallel models * "gguf"
                        will load weights from GGUF format files (details
                        specified in https://github.com/ggml-
                        org/ggml/blob/master/docs/gguf.md). * "mistral" will
                        load weights from consolidated safetensors files used
                        by Mistral models. (default: auto)
  --logits-processor-pattern LOGITS_PROCESSOR_PATTERN
                        Optional regex pattern specifying valid logits
                        processor qualified names that can be passed with the
                        `logits_processors` extra completion argument.
                        Defaults to None, which allows no processors.
                        (default: None)
  --long-lora-scaling-factors LONG_LORA_SCALING_FACTORS
                        Specify multiple scaling factors (which can be
                        different from base model scaling factor - see eg.
                        Long LoRA) to allow for multiple LoRA adapters trained
                        with those scaling factors to be used at the same
                        time. If not specified, only adapters trained with the
                        base model scaling factor are allowed. (default: None)
  --long-prefill-token-threshold LONG_PREFILL_TOKEN_THRESHOLD
                        For chunked prefill, a request is considered long if
                        the prompt is longer than this number of tokens.
                        (default: 0)
  --lora-dtype {auto,float16,bfloat16}
                        Data type for LoRA. If auto, will default to base
                        model dtype. (default: auto)
  --lora-extra-vocab-size LORA_EXTRA_VOCAB_SIZE
                        Maximum size of extra vocabulary that can be present
                        in a LoRA adapter (added to the base model
                        vocabulary). (default: 256)
  --lora-modules LORA_MODULES [LORA_MODULES ...]
                        LoRA module configurations in either 'name=path'
                        formator JSON format. Example (old format):
                        ``'name=path'`` Example (new format): ``{"name":
                        "name", "path": "lora_path", "base_model_name":
                        "id"}`` (default: None)
  --max-cpu-loras MAX_CPU_LORAS
                        Maximum number of LoRAs to store in CPU memory. Must
                        be >= than max_loras. (default: None)
  --max-log-len MAX_LOG_LEN
                        Max number of prompt characters or prompt ID numbers
                        being printed in log. The default of None means
                        unlimited. (default: None)
  --max-logprobs MAX_LOGPROBS
                        Max number of log probs to return logprobs is
                        specified in SamplingParams. (default: 20)
  --max-long-partial-prefills MAX_LONG_PARTIAL_PREFILLS
                        For chunked prefill, the maximum number of prompts
                        longer than --long-prefill-token-threshold that will
                        be prefilled concurrently. Setting this less than
                        --max-num-partial-prefills will allow shorter prompts
                        to jump the queue in front of longer prompts in some
                        cases, improving latency. (default: 1)
  --max-lora-rank MAX_LORA_RANK
                        Max LoRA rank. (default: 16)
  --max-loras MAX_LORAS
                        Max number of LoRAs in a single batch. (default: 1)
  --max-model-len MAX_MODEL_LEN
                        Model context length. If unspecified, will be
                        automatically derived from the model config. (default:
                        None)
  --max-num-batched-tokens MAX_NUM_BATCHED_TOKENS
                        Maximum number of batched tokens per iteration.
                        (default: None)
  --max-num-partial-prefills MAX_NUM_PARTIAL_PREFILLS
                        For chunked prefill, the max number of concurrent
                        partial prefills. (default: 1)
  --max-num-seqs MAX_NUM_SEQS
                        Maximum number of sequences per iteration. (default:
                        None)
  --max-parallel-loading-workers MAX_PARALLEL_LOADING_WORKERS
                        Load model sequentially in multiple batches, to avoid
                        RAM OOM when using tensor parallel and large models.
                        (default: None)
  --max-prompt-adapter-token MAX_PROMPT_ADAPTER_TOKEN
                        Max number of PromptAdapters tokens (default: 0)
  --max-prompt-adapters MAX_PROMPT_ADAPTERS
                        Max number of PromptAdapters in a batch. (default: 1)
  --max-seq-len-to-capture MAX_SEQ_LEN_TO_CAPTURE
                        Maximum sequence length covered by CUDA graphs. When a
                        sequence has context length larger than this, we fall
                        back to eager mode. Additionally for encoder-decoder
                        models, if the sequence length of the encoder input is
                        larger than this, we fall back to the eager mode.
                        (default: 8192)
  --middleware MIDDLEWARE
                        Additional ASGI middleware to apply to the app. We
                        accept multiple --middleware arguments. The value
                        should be an import path. If a function is provided,
                        vLLM will add it to the server using
                        ``@app.middleware('http')``. If a class is provided,
                        vLLM will add it to the server using
                        ``app.add_middleware()``. (default: [])
  --mm-processor-kwargs MM_PROCESSOR_KWARGS
                        Overrides for the multimodal input mapping/processing,
                        e.g., image processor. For example: ``{"num_crops":
                        4}``. (default: None)
  --model MODEL         Name or path of the huggingface model to use.
                        (default: facebook/opt-125m)
  --model-impl {auto,vllm,transformers}
                        Which implementation of the model to use. * "auto"
                        will try to use the vLLM implementation if it exists
                        and fall back to the Transformers implementation if no
                        vLLM implementation is available. * "vllm" will use
                        the vLLM model implementation. * "transformers" will
                        use the Transformers model implementation. (default:
                        auto)
  --model-loader-extra-config MODEL_LOADER_EXTRA_CONFIG
                        Extra config for model loader. This will be passed to
                        the model loader corresponding to the chosen
                        load_format. This should be a JSON string that will be
                        parsed into a dictionary. (default: None)
  --multi-step-stream-outputs [MULTI_STEP_STREAM_OUTPUTS]
                        If False, then multi-step will stream outputs at the
                        end of all steps (default: True)
  --num-gpu-blocks-override NUM_GPU_BLOCKS_OVERRIDE
                        If specified, ignore GPU profiling result and use this
                        number of GPU blocks. Used for testing preemption.
                        (default: None)
  --num-lookahead-slots NUM_LOOKAHEAD_SLOTS
                        Experimental scheduling config necessary for
                        speculative decoding. This will be replaced by
                        speculative config in the future; it is present to
                        enable correctness tests until then. (default: 0)
  --num-scheduler-steps NUM_SCHEDULER_STEPS
                        Maximum number of forward steps per scheduler call.
                        (default: 1)
  --otlp-traces-endpoint OTLP_TRACES_ENDPOINT
                        Target URL to which OpenTelemetry traces will be sent.
                        (default: None)
  --override-generation-config OVERRIDE_GENERATION_CONFIG
                        Overrides or sets generation config in JSON format.
                        e.g. ``{"temperature": 0.5}``. If used with
                        --generation-config=auto, the override parameters will
                        be merged with the default config from the model. If
                        generation-config is None, only the override
                        parameters are used. (default: None)
  --override-neuron-config OVERRIDE_NEURON_CONFIG
                        Override or set neuron device configuration. e.g.
                        ``{"cast_logits_dtype": "bloat16"}``. (default: None)
  --override-pooler-config OVERRIDE_POOLER_CONFIG
                        Override or set the pooling method for pooling models.
                        e.g. ``{"pooling_type": "mean", "normalize": false}``.
                        (default: None)
  --pipeline-parallel-size PIPELINE_PARALLEL_SIZE, -pp PIPELINE_PARALLEL_SIZE
                        Number of pipeline stages. (default: 1)
  --port PORT           Port number. (default: 8000)
  --preemption-mode PREEMPTION_MODE
                        If 'recompute', the engine performs preemption by
                        recomputing; If 'swap', the engine performs preemption
                        by block swapping. (default: None)
  --prefix-caching-hash-algo {builtin,sha256}
                        Set the hash algorithm for prefix caching. Options are
                        'builtin' (Python's built-in hash) or 'sha256'
                        (collision resistant but with certain overheads).
                        (default: builtin)
  --prompt-adapters PROMPT_ADAPTERS [PROMPT_ADAPTERS ...]
                        Prompt adapter configurations in the format name=path.
                        Multiple adapters can be specified. (default: None)
  --qlora-adapter-name-or-path QLORA_ADAPTER_NAME_OR_PATH
                        Name or path of the QLoRA adapter. (default: None)
  --quantization {aqlm,awq,deepspeedfp,tpu_int8,fp8,ptpc_fp8,fbgemm_fp8,modelopt,nvfp4,marlin,gguf,gptq_marlin_24,gptq_marlin,awq_marlin,gptq,compressed-tensors,bitsandbytes,qqq,hqq,experts_int8,neuron_quant,ipex,quark,moe_wna16,None}, -q {aqlm,awq,deepspeedfp,tpu_int8,fp8,ptpc_fp8,fbgemm_fp8,modelopt,nvfp4,marlin,gguf,gptq_marlin_24,gptq_marlin,awq_marlin,gptq,compressed-tensors,bitsandbytes,qqq,hqq,experts_int8,neuron_quant,ipex,quark,moe_wna16,None}
                        Method used to quantize the weights. If None, we first
                        check the `quantization_config` attribute in the model
                        config file. If that is None, we assume the model
                        weights are not quantized and use `dtype` to determine
                        the data type of the weights. (default: None)
  --ray-workers-use-nsight
                        If specified, use nsight to profile Ray workers.
                        (default: False)
  --reasoning-parser {deepseek_r1,granite}
                        Select the reasoning parser depending on the model
                        that you're using. This is used to parse the reasoning
                        content into OpenAI API format. Required for
                        ``--enable-reasoning``. (default: None)
  --response-role RESPONSE_ROLE
                        The role name to return if
                        ``request.add_generation_prompt=true``. (default:
                        assistant)
  --return-tokens-as-token-ids
                        When ``--max-logprobs`` is specified, represents
                        single tokens as strings of the form
                        'token_id:{token_id}' so that tokens that are not
                        JSON-encodable can be identified. (default: False)
  --revision REVISION   The specific model version to use. It can be a branch
                        name, a tag name, or a commit id. If unspecified, will
                        use the default version. (default: None)
  --root-path ROOT_PATH
                        FastAPI root_path when app is behind a path based
                        routing proxy. (default: None)
  --rope-scaling ROPE_SCALING
                        RoPE scaling configuration in JSON format. For
                        example, ``{"rope_type":"dynamic","factor":2.0}``
                        (default: None)
  --rope-theta ROPE_THETA
                        RoPE theta. Use with `rope_scaling`. In some cases,
                        changing the RoPE theta improves the performance of
                        the scaled model. (default: None)
  --scheduler-cls SCHEDULER_CLS
                        The scheduler class to use.
                        "vllm.core.scheduler.Scheduler" is the default
                        scheduler. Can be a class directly or the path to a
                        class of form "mod.custom_class". (default:
                        vllm.core.scheduler.Scheduler)
  --scheduler-delay-factor SCHEDULER_DELAY_FACTOR
                        Apply a delay (of delay factor multiplied by previous
                        prompt latency) before scheduling next prompt.
                        (default: 0.0)
  --scheduling-policy {fcfs,priority}
                        The scheduling policy to use. "fcfs" (first come first
                        served, i.e. requests are handled in order of arrival;
                        default) or "priority" (requests are handled based on
                        given priority (lower value means earlier handling)
                        and time of arrival deciding any ties). (default:
                        fcfs)
  --seed SEED           Random seed for operations. (default: None)
  --served-model-name SERVED_MODEL_NAME [SERVED_MODEL_NAME ...]
                        The model name(s) used in the API. If multiple names
                        are provided, the server will respond to any of the
                        provided names. The model name in the model field of a
                        response will be the first name in this list. If not
                        specified, the model name will be the same as the
                        ``--model`` argument. Noted that this name(s) will
                        also be used in `model_name` tag content of prometheus
                        metrics, if multiple names provided, metrics tag will
                        take the first one. (default: None)
  --show-hidden-metrics-for-version SHOW_HIDDEN_METRICS_FOR_VERSION
                        Enable deprecated Prometheus metrics that have been
                        hidden since the specified version. For example, if a
                        previously deprecated metric has been hidden since the
                        v0.7.0 release, you use --show-hidden-metrics-for-
                        version=0.7 as a temporary escape hatch while you
                        migrate to new metrics. The metric is likely to be
                        removed completely in an upcoming release. (default:
                        None)
  --skip-tokenizer-init
                        Skip initialization of tokenizer and detokenizer.
                        Expects valid prompt_token_ids and None for prompt
                        from the input. The generated output will contain
                        token ids. (default: False)
  --speculative-config SPECULATIVE_CONFIG
                        The configurations for speculative decoding. Should be
                        a JSON string. (default: None)
  --ssl-ca-certs SSL_CA_CERTS
                        The CA certificates file. (default: None)
  --ssl-cert-reqs SSL_CERT_REQS
                        Whether client certificate is required (see stdlib ssl
                        module's). (default: 0)
  --ssl-certfile SSL_CERTFILE
                        The file path to the SSL cert file. (default: None)
  --ssl-keyfile SSL_KEYFILE
                        The file path to the SSL key file. (default: None)
  --swap-space SWAP_SPACE
                        CPU swap space size (GiB) per GPU. (default: 4)
  --task {auto,generate,embedding,embed,classify,score,reward,transcription}
                        The task to use the model for. Each vLLM instance only
                        supports one task, even if the same model can be used
                        for multiple tasks. When the model only supports one
                        task, ``"auto"`` can be used to select it; otherwise,
                        you must specify explicitly which task to use.
                        (default: auto)
  --tensor-parallel-size TENSOR_PARALLEL_SIZE, -tp TENSOR_PARALLEL_SIZE
                        Number of tensor parallel replicas. (default: 1)
  --tokenizer TOKENIZER
                        Name or path of the huggingface tokenizer to use. If
                        unspecified, model name or path will be used.
                        (default: None)
  --tokenizer-mode {auto,slow,mistral,custom}
                        The tokenizer mode. * "auto" will use the fast
                        tokenizer if available. * "slow" will always use the
                        slow tokenizer. * "mistral" will always use the
                        `mistral_common` tokenizer. * "custom" will use
                        --tokenizer to select the preregistered tokenizer.
                        (default: auto)
  --tokenizer-pool-extra-config TOKENIZER_POOL_EXTRA_CONFIG
                        Extra config for tokenizer pool. This should be a JSON
                        string that will be parsed into a dictionary. Ignored
                        if tokenizer_pool_size is 0. (default: None)
  --tokenizer-pool-size TOKENIZER_POOL_SIZE
                        Size of tokenizer pool to use for asynchronous
                        tokenization. If 0, will use synchronous tokenization.
                        (default: 0)
  --tokenizer-pool-type TOKENIZER_POOL_TYPE
                        Type of tokenizer pool to use for asynchronous
                        tokenization. Ignored if tokenizer_pool_size is 0.
                        (default: ray)
  --tokenizer-revision TOKENIZER_REVISION
                        Revision of the huggingface tokenizer to use. It can
                        be a branch name, a tag name, or a commit id. If
                        unspecified, will use the default version. (default:
                        None)
  --tool-call-parser {granite-20b-fc,granite,hermes,internlm,jamba,llama3_json,mistral,phi4_mini_json,pythonic} or name registered in --tool-parser-plugin
                        Select the tool call parser depending on the model
                        that you're using. This is used to parse the model-
                        generated tool call into OpenAI API format. Required
                        for ``--enable-auto-tool-choice``. (default: None)
  --tool-parser-plugin TOOL_PARSER_PLUGIN
                        Special the tool parser plugin write to parse the
                        model-generated tool into OpenAI API format, the name
                        register in this plugin can be used in ``--tool-call-
                        parser``. (default: )
  --trust-remote-code   Trust remote code from huggingface. (default: False)
  --use-tqdm-on-load, --no-use-tqdm-on-load
                        Whether to enable/disable progress bar when loading
                        model weights. (default: True)
  --use-v2-block-manager
                        [DEPRECATED] block manager v1 has been removed and
                        SelfAttnBlockSpaceManager (i.e. block manager v2) is
                        now the default. Setting this flag to True or False
                        has no effect on vLLM behavior. (default: True)
  --uvicorn-log-level {debug,info,warning,error,critical,trace}
                        Log level for uvicorn. (default: info)
  --worker-cls WORKER_CLS
                        The worker class to use for distributed execution.
                        (default: auto)
  --worker-extension-cls WORKER_EXTENSION_CLS
                        The worker extension class on top of the worker cls,
                        it is useful if you just want to add new functions to
                        the worker class without changing the existing
                        functions. (default: )
  -h, --help            show this help message and exit
