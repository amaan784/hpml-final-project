# Qwen2.5-VL W4A16 via llmcompressor 0.10 -> compressed-tensors dump for vLLM 0.19+.
# Modes: w4a16_domain (domain text), w4a16_generic (ultrachat).

import argparse
import os
import sys
import sysconfig
from pathlib import Path

# Disable wandb noise inside oneshot (llmcompressor pulls wandb at import).
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")

# Reduce CUDA fragmentation. Qwen2.5-VL's MLP intermediate (18944) generates a
# 1.43 GB Hessian for the down_proj Linear. On a 22 GiB L4 alongside the
# 15 GB FP16 model, fragmentation triggers OOM during ``H[perm][:, perm]``
# permutation. Expandable segments let the allocator grow into freed blocks
# instead of requiring contiguous regions.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _ensure_transformers_save_pretrained_patch() -> None:
    """Patch transformers 4.57's offloaded-save module_map bug before import.

    This must run before importing transformers.modeling_utils. The loaded
    save_pretrained bytecode otherwise keeps the buggy `if module_map:` branch,
    which crashes on parameter-only Qwen/LLaVA vision attributes during
    save_pretrained(save_compressed=True).
    """
    site_paths = [sysconfig.get_paths().get(k) for k in ("purelib", "platlib")]
    # step through the batch one entry at a time
    for base in filter(None, site_paths):
        path = Path(base) / "transformers" / "modeling_utils.py"
        if path.exists():
            break
    else:
        raise RuntimeError("could not find transformers/modeling_utils.py to patch")

    text = path.read_text()

    if "if module_map and False:" in text:
        print(f"==> transformers save_pretrained patch already present: {path}")
        return

    if "if module_map:" not in text:
        raise RuntimeError(
            "transformers modeling_utils.py does not contain the expected "
            "`if module_map:` guard; inspect save_pretrained before quantizing."
        )
    path.write_text(text.replace("if module_map:", "if module_map and False:", 1))
    print(f"==> patched transformers save_pretrained module_map guard: {path}")


def _build_dataset(mode: str, num_samples: int):
    """Return a ``datasets.Dataset`` with a single ``text`` column."""
    from datasets import Dataset
    from benchmark.calibration import build_domain_corpus, build_generic_corpus

    if mode == "w4a16_domain":
        texts = build_domain_corpus(num_samples=num_samples)
        print(f"==> DOMAIN calibration: {len(texts)} substation texts")
    elif mode == "w4a16_generic":
        texts = build_generic_corpus(num_samples=num_samples)
        print(f"==> GENERIC calibration: {len(texts)} ultrachat samples")
    else:
        raise ValueError(f"Unknown mode {mode!r}")
    return Dataset.from_dict({"text": texts})


def _patch_qwen_vl_for_fx_tracing() -> None:
    """Make Qwen2.5-VL attention survivable under llmcompressor's FX tracer.

    llmcompressor 0.10's sequential pipeline FX-traces each subgraph between
    its ``sequential_targets``. ``Qwen2_5_VLAttention.forward`` does:

        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling['mrope_section']
        )

    The tuple-unpack on a Proxy object trips torch.fx with
    "Proxy object cannot be iterated". Setting
    ``sequential_targets=["Qwen2_5_VLDecoderLayer"]`` does NOT prevent this on
    transformers 4.57.6 + llmcompressor 0.10.0.2 -- the tracer still descends
    into attention internals because the autowrap heuristic only catches
    *some* untraceable patterns (it autowraps the cache-update branch but
    not the RoPE call).

    The fix has two parts:

    1. Register the RoPE helper in torch.fx's wrap registry at the module level
       where it's defined, so symbolic tracing treats it as a leaf op.
    2. Replace ``Qwen2_5_VLAttention.forward`` with an equivalent implementation
       that indexes the helper's tuple-like output as ``rotary[0]`` /
       ``rotary[1]``. Python tuple unpack tries to iterate a Proxy; indexing
       emits normal ``operator.getitem`` nodes and is FX-traceable.

    The runtime math is unchanged.

    Idempotent: safe to call multiple times.
    """
    # wrap risky IO or RPC so we can surface a useful failure
    try:
        import torch.fx._symbolic_trace as _fxst
        from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as _mod
    except ImportError:
        print("==> WARNING: could not patch Qwen2.5-VL FX tracing "
              "(transformers / torch.fx not installed)")
        return

    name = "apply_multimodal_rotary_pos_emb"

    if not hasattr(_mod, name):
        print(f"==> WARNING: {name} not found in modeling_qwen2_5_vl "
              "(transformers version mismatch?). Skipping FX patch.")
        return

    key = (id(_mod.__dict__), name)

    if key in getattr(_fxst, "_wrapped_fns_to_patch", {}):
        print(f"==> patched: {name} already registered as torch.fx leaf op")
    else:
        _fxst._wrapped_fns_to_patch[key] = _mod.__dict__
        print(f"==> patched: {name} registered as torch.fx leaf op")

    attention_cls = getattr(_mod, "Qwen2_5_VLAttention", None)

    if attention_cls is None:
        print("==> WARNING: Qwen2_5_VLAttention not found; skipping forward patch.")
        return

    if getattr(attention_cls.forward, "_assetops_fx_safe_forward", False):
        return

    def _fx_safe_forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        if past_key_values is None and "past_key_value" in kwargs:
            past_key_values = kwargs.pop("past_key_value")

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        rotary = _mod.apply_multimodal_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            self.rope_scaling["mrope_section"],
        )
        query_states = rotary[0]
        key_states = rotary[1]

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        attention_interface = _mod.eager_attention_forward

        if self.config._attn_implementation != "eager":
            attention_interface = _mod.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            position_ids=position_ids,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    _fx_safe_forward._assetops_fx_safe_forward = True
    attention_cls.forward = _fx_safe_forward
    print("==> patched: Qwen2_5_VLAttention.forward is FX-safe")


def _resolve_decoder_layer_class_name() -> str:
    """Find the actual text-decoder-layer class name in this transformers build.

    transformers 4.57+ renamed several Qwen2.5-VL inner classes (the text
    tower split out as ``Qwen2_5_VLTextDecoderLayer`` in some builds). Search
    the modeling module so we set ``sequential_targets`` to whatever's
    actually defined here -- a missing target name silently disables the
    boundary and the tracer walks the entire model.
    """
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as _mod

    # repeat for every element we need to touch
    for candidate in (
        "Qwen2_5_VLDecoderLayer",
        "Qwen2_5_VLTextDecoderLayer",
        "Qwen2VLDecoderLayer",
    ):
        if hasattr(_mod, candidate):
            return candidate
    # Fall back: scan for anything ending in DecoderLayer.
    matches = [n for n in dir(_mod) if n.endswith("DecoderLayer") and not n.startswith("_")]

    if matches:
        return matches[0]
    raise RuntimeError(
        "No DecoderLayer class found in transformers.models.qwen2_5_vl.modeling_qwen2_5_vl. "
        "transformers may be too old or the module has been restructured."
    )


def _build_recipe():
    """W4A16 GPTQ recipe. Vision tower stays FP16 (re:visual.*).

    ``sequential_targets`` is passed directly to ``oneshot`` below. Keeping it
    out of the modifier avoids deprecated llmcompressor behavior where the
    target may be ignored or warning-only.
    """
    from llmcompressor.modifiers.quantization import GPTQModifier

    return GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head", "re:visual.*", "re:model.visual.*"],
    )


def _save_with_offload_dance(model, processor, tokenizer, out_dir: str) -> None:
    """Save quantized weights via the dispatch_model + remove_hook dance.

    Without this, llmcompressor's calibration leaves accelerate offload hooks
    on the model tree, and ``save_pretrained``'s offload-aware branch crashes
    on parameter-only attributes (e.g. ``image_newline``, ``class_embedding``).

    Going through ``save_pretrained(save_compressed=True)`` is mandatory --
    that's the hook that fires llmcompressor's compression callbacks to pack
    weights into real INT4 storage. Bypassing it dumps FP16-dequantized
    weights instead.
    """
    print("   dispatch_model: consolidating offloaded modules ...")
    # keep the happy path obvious by catching failures here
    try:
        from compressed_tensors.offload import dispatch_model
    except ImportError:
        from accelerate import dispatch_model  # type: ignore[no-redef]
    dispatch_model(model)

    print("   remove_hook_from_module: stripping accelerate offload hooks ...")
    from accelerate.hooks import remove_hook_from_module
    remove_hook_from_module(model, recurse=True)

    meta_params = [name for name, param in model.named_parameters() if param.device.type == "meta"]

    if meta_params:
        preview = ", ".join(meta_params[:8])
        raise RuntimeError(
            "dispatch_model did not materialize all parameters before save; "
            f"{len(meta_params)} parameters are still on meta, e.g. {preview}"
        )

    # Transformers decides whether to use its offloaded-save path from the
    # hf_device_map attribute, not from the actual tensor devices. After
    # dispatch_model() + remove_hook_from_module(), the tensors are materialized,
    # but Qwen2.5-VL can still carry a stale device map with cpu/disk entries.
    # That path crashes on vision keys such as visual.patch_embed.proj.weight.
    for module in model.modules():
        if hasattr(module, "hf_device_map"):
            try:
                delattr(module, "hf_device_map")
            except AttributeError:
                module.hf_device_map = None

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, save_compressed=True)
    print("   save_pretrained completed (compression hooks fired -> packed INT4)")

    tokenizer.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["w4a16_domain", "w4a16_generic"], required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--model-id",
        default=os.environ.get("AWQ_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct"),
    )
    p.add_argument("--num-samples", type=int, default=128)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument(
        "--pipeline",
        choices=["sequential", "basic"],
        default="basic",
        help=(
            "llmcompressor pipeline. "
            "'basic' (default): single calibration pass that bypasses "
            "llmcompressor's FX tracer and avoids the Qwen2.5-VL "
            "`Proxy object cannot be iterated` failure. "
            "'sequential': subgraph-by-subgraph, lower peak memory, "
            "requires FX-traceable model (the multimodal-RoPE leaf-op patch "
            "addresses the main issue, but Qwen2_5_VLAttention.forward also "
            "has **kwargs forwarding that FX can't trace -- if you still hit "
            "'Proxy object cannot be iterated', use the default 'basic')."
        ),
    )
    args = p.parse_args()

    _ensure_transformers_save_pretrained_patch()

    import torch  # noqa: F401  (forces CUDA init before transformers loads)
    from transformers import AutoProcessor, AutoTokenizer, AutoModelForVision2Seq
    import llmcompressor
    from llmcompressor import oneshot

    print(
        "==> versions: "
        f"torch={torch.__version__} "
        f"llmcompressor={getattr(llmcompressor, '__version__', 'unknown')}"
    )

    # MUST run BEFORE oneshot() so the FX tracer picks up the leaf-op
    # registration when it traces Qwen2_5_VLAttention.forward. Otherwise the
    # tracer dies on the tuple-unpack of apply_multimodal_rotary_pos_emb's
    # return value with "Proxy object cannot be iterated".
    _patch_qwen_vl_for_fx_tracing()

    print(f"==> Loading {args.model_id} (FP16) ...")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        dtype="float16",
        device_map="auto",
        attn_implementation="eager",  # avoids RoPE/SDPA mismatch in calibration
    )
    model.config.use_cache = False

    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False

    calib_ds = _build_dataset(args.mode, args.num_samples)
    recipe = _build_recipe()
    decoder_class = _resolve_decoder_layer_class_name()
    print(f"==> sequential_targets = [{decoder_class!r}]")

    print(f"==> oneshot quantization (W4A16, pipeline={args.pipeline})")
    oneshot_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        dataset=calib_ds,
        recipe=recipe,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=len(calib_ds),
        sequential_targets=[decoder_class],
    )
    # `pipeline="basic"` is the fix for this Qwen FX tracing failure. If this
    # llmcompressor build cannot accept the kwarg, do not continue into its
    # default sequential path -- that is the path that raises
    # "Proxy object cannot be iterated".
    import inspect as _inspect

    if "pipeline" in _inspect.signature(oneshot).parameters:
        oneshot_kwargs["pipeline"] = args.pipeline
    else:
        raise RuntimeError(
            "Installed llmcompressor is too old for the Qwen2.5-VL hotfix: "
            "oneshot() does not accept pipeline=. Install "
            "llmcompressor==0.10.0.2, compressed-tensors==0.14.0.1, "
            "and vllm==0.19.0 in ~/HPML-AssetOpsBench/.venv."
        )
    oneshot(**oneshot_kwargs)

    print(f"==> Saving full Qwen2.5-VL wrapper (with quantized LM) -> {args.out_dir}")
    _save_with_offload_dance(model, processor, tokenizer, args.out_dir)

    import json
    cfg_path = Path(args.out_dir) / "config.json"
    qc = {}

    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        qc = cfg.get("quantization_config", {})
        print(
            f"   config.json quantization_config: present={bool(qc)} "
            f"method={qc.get('quant_method', 'n/a')} "
            f"format={qc.get('format', 'n/a')}"
        )

    # Log a metadata-only W&B Artifact (config + size manifest, NOT the
    # 5 GB safetensors weights -- those live under models/ on the VM).
    # Re-enable wandb just for this step (was disabled at script start).
    os.environ["WANDB_DISABLED"] = "false"
    os.environ["WANDB_MODE"] = "online"
    # keep the happy path obvious by catching failures here
    try:
        sys.path.insert(0, str(_REPO))
        from benchmark.wandb_logger import log_checkpoint_artifact
        artifact_url = log_checkpoint_artifact(
            name=Path(args.out_dir).name,
            out_dir=args.out_dir,
            metadata={
                "model_id": args.model_id,
                "mode": args.mode,
                "num_calibration_samples": args.num_samples,
                "max_seq_len": args.max_seq_len,
                "sequential_targets": decoder_class,
                "scheme": "W4A16",
                "quant_method": qc.get("quant_method", "compressed-tensors"),
                "format": qc.get("format", "pack-quantized"),
            },
            description=f"Qwen2.5-VL-7B {args.mode} (W4A16 GPTQ)",
        )
        if artifact_url:
            print(f"==> wandb artifact run: {artifact_url}")
    except Exception as exc:  # noqa: BLE001
        print(f"   (wandb artifact logging skipped: {exc})")

    print("==> Done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

