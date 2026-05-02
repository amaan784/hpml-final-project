# PyTorch Profiler trace of the Qwen2.5-VL vision encoder in isolation.
# Run on the GCP VM with the model weights cached:
#   python benchmark/profile_vision_encoder.py
#   tensorboard --logdir tb

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from servers.vision import image_loader


def main():
    # Lightweight CLI wrapper — heavy torch imports delayed until argparse finishes.
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"))
    p.add_argument("--image", default="hf://substation/train/0")
    p.add_argument("--tb-dir", default="./tb")
    args = p.parse_args()

    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print(f"==> Loading {args.model} (FP16) ...")
    # Processor aligns images with whatever normalization the HF config expects.
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()

    img = image_loader.load_image(args.image)
    # Build a multimodal tensor batch exactly like downstream inference does.
    inputs = proc(images=img, text="dummy", return_tensors="pt").to("cuda")

    # TensorBoard hook captures CUDA kernels for visual encoder only.
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(args.tb_dir),
        record_shapes=True,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            # Qwen2.5-VL exposes the vision tower at .visual. fall back to
            # .vision_tower for other layouts
            visual = getattr(model, "visual", None) or getattr(model, "vision_tower", None)
            if visual is None:
                raise RuntimeError("Could not locate vision tower (.visual / .vision_tower)")
            _ = visual(inputs["pixel_values"])
            torch.cuda.synchronize()
        prof.step()

    print("==> Trace written to", args.tb_dir)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    return 0

if __name__ == "__main__":
    sys.exit(main())

