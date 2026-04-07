# HPML Final Project - VLM Comparison on Industrial Thermal Images

Comparing two Vision-Language Models on thermal fault detection for induction motors.

## Models

| Model | Parameters | Quantization |
|---|---|---|
| [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | 7B | 4-bit (bitsandbytes) |
| [Llama-3.2-Vision-11B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) | 11B | 4-bit (bitsandbytes) |

## Dataset

Thermal images of induction motors from the [Mendeley dataset](https://data.mendeley.com/datasets/m4sbt8hbvk/3) (11 fault categories). We evaluate on 5 categories:

1. Healthy motor
2. Stator short circuit fault
3. Stuck rotor
4. Cooling fan failure
5. Bearing fault (or another category of your choice)

## Evaluation

Each model is asked 3 questions per image:

1. **Fault identification** - Is the equipment operating normally or is there a fault?
2. **Hotspot detection** - Are there abnormal temperature patterns? Where and why?
3. **Maintenance recommendation** - Should maintenance action be taken?

Responses are scored on a 1-5 rubric and compared side-by-side.

## Hardware

- **Target GPU:** NVIDIA L4 (24 GB VRAM)
- Both models loaded in 4-bit quantization via `bitsandbytes`
- Models are loaded one at a time to fit in VRAM

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/<your-username>/hpml-final-project.git
   cd hpml-final-project
   ```

2. **Download thermal images** from the [Mendeley dataset](https://data.mendeley.com/datasets/m4sbt8hbvk/3) and place them in a `thermal_images/` folder (or update the path in the notebook config cell).

3. **Accept the Llama license** at https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct

4. **Install dependencies**
   ```bash
   pip install transformers accelerate torch qwen-vl-utils bitsandbytes pillow
   ```

5. **Run the notebook**
   ```
   vlm_thermal_comparison.ipynb
   ```
   Log in to HuggingFace when prompted, then run cells sequentially.

## Project Structure

```
├── vlm_thermal_comparison.ipynb   # Main experiment notebook
├── thermal_images/                # Downloaded test images (not committed)
├── qwen_results.json              # Qwen model outputs (generated)
├── llama_results.json             # Llama model outputs (generated)
├── vlm_comparison_scores.csv      # Scoring table (generated)
└── README.md
```

## License

See [LICENSE](LICENSE).
