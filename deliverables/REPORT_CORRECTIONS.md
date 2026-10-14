# Corrections to the submitted project report

The report's LaTeX source is not in this repository. This file lists every
correction found after submission, keyed to the section of
`HPML Team 23 Project Report.pdf`, with the submitted text and paste-ready
replacement text for the Overleaf source. The README has already been updated
to match.

The corrections come from an external review plus checks against the
committed results (`results/summary.csv`, `results/llm_judge.csv`). Numbers
marked *provisional* will be replaced by the output of
`python scripts/report_numbers.py` after the three-pass re-run.

---

## 1. §III.B — the benchmark does not run through the ReAct agent

**Why:** `benchmark/run_vlm_benchmark.py` calls the vision tool directly, one
`analyze_image` call per scenario, and never runs the planner. The submitted
text led a reviewer to explain results through "agent turns" that do not exist.

**Submitted:**
> The results of the benchmark experiments reported here were performed with
> the agent invoked by our in-house ReAct [23] orchestrator vs. the upstream
> MetaAgent or AgentHive registry in order to isolate inference-optimization
> measurements from noise created at the orchestrator level.

**Replace with:**
> The benchmark experiments reported here bypass agent orchestration: the
> harness (`benchmark/run_vlm_benchmark.py`) calls the Visual Inspection
> Agent's `analyze_image` tool directly, once per scenario, which isolates
> inference-optimization measurements from orchestrator noise. Reported latency
> and accuracy therefore measure the VLM serving path, not multi-turn agent
> behavior. The in-house ReAct [23] orchestrator uses the same tools but is not
> in the measured path.

## 2. §V.B — same correction

**Submitted:**
> For the benchmark experiments that we report on here, the agent was invoked
> directly through our in-house ReAct [23] orchestrator rather than through the
> upstream registry which simplifies per-scenario timing and removes the
> WatsonX dependency entirely from the application.

**Replace with:**
> For the benchmark experiments reported here, the harness called the vision
> tool directly (one `analyze_image` call per scenario) rather than through any
> orchestrator, which simplifies per-scenario timing and removes the WatsonX
> dependency. Evaluating the full agent loop is future work (Section V-C).

## 3. §III.C — calibration corpus, L2 defaults, and L3 scope

**Why:** the domain corpus is 40 short substation sentences repeated to 128
samples, not 128 prompts. Under vLLM 0.19's V1 engine, prefix caching and
chunked prefill are on by default, so L0 already used them. The 512 px cap
leaves half the images unchanged.

**Submitted:**
> L1d employs 128 application-specific inspection prompts; while L1g operates
> using 128 random-based ultrachat_200k samples.

**Replace with:**
> L1d uses a domain corpus of 40 unique short substation and transformer
> sentences, cycled to 128 samples (a few thousand tokens in total). It
> contains no pump, motor, or turbine text, and a few sentences describe the
> multi-panel layout of the transformer test images. L1g uses 128
> ultrachat_200k conversations of up to 8,000 characters each. Because the
> corpora differ in size and variety as well as topic, L1d-versus-L1g
> comparisons do not isolate the effect of topic. Calibration is text-only.

**Submitted:**
> The third layer (L2) includes support for the use of vLLM serving options
> associated with prefix caching operations, chunked prefill operation, FP8-KV
> cache use, and continuous batching (bundled as full_bundle).

**Replace with:**
> The third layer (L2) enables vLLM's prefix caching, chunked prefill, and FP8
> KV cache with a 0.90 GPU-memory budget (bundled as full_bundle). Under vLLM
> 0.19's V1 engine, prefix caching, chunked prefill, and continuous batching
> are already on by default, so the L0 baseline ran with them as well; the
> bundle's correctness-relevant change relative to L0 is the FP8 KV cache.

**Submitted:**
> The fourth layer (L3) runs all incoming images at 512 pixels prior to
> encoding.

**Replace with:**
> The fourth layer (L3) caps each image's longest side at 512 pixels before
> encoding. Images already at or below 512 px (all 5 pump and 6 motor
> scenarios) are unchanged, so L3 alters the input for only 11 of the 22
> scenarios.

## 4. §III.D — judge methodology (apply after the re-run)

**Why:** the repository holds only one of the two judge passes behind Tables
II and III, and repeated passes disagree. The re-run uses a pinned snapshot,
a fixed seed per pass, and three passes, and it counts empty answers as fails.

**Submitted:**
> ... was used for each of the 22 scenarios evaluated by up to two judges each
> (maximum denominator of 44).

**Replace with (after the re-run):**
> ... judged each of the 22 scenarios in three independent passes
> (gpt-4o-mini-2024-07-18, pinned snapshot, fixed seed per pass). We report
> the mean ± standard deviation of the pass rate across passes. An empty answer
> is scored as a fail rather than skipped.

## 5. Abstract

**Submitted:**
> Additionally, the LLM-judge pass rate for Qwen improved from 21 out of 44
> (47.7%) to 36 out of 44 (81.8%), representing an absolute improvement of 34
> percentage points.

**Replace with:**
> The LLM-judge pass rate for Qwen was also higher (36/44 vs 21/44), which we
> treat as provisional given 22 scenarios and run-to-run judge variance.

**Submitted:**
> Qwen's average response time decreased to 2,000 ms (4.7× speedup), but all
> responses returned were off-topic or empty (0 out of 44 judge-pass)

**Replace with:**
> Qwen's mean latency fell to 2,000 ms only because generation collapsed into
> degenerate output of a few tokens (0 of 44 judge-pass)

## 6. §I contribution 4

**Why:** the results do not come from CodaBench, and the accuracy gain is
provisional.

**Submitted:**
> Clear quantitative results showing that AWQ W4A16 using domain-specific
> calibration data has the largest performance gain based on CodaBench's
> results and that this variant provides the only performance improvement of
> judge accuracy on Qwen

**Replace with:**
> Quantitative results showing that AWQ W4A16 with domain calibration gives
> the largest latency gain on Qwen (1.99×), with a provisional judge-accuracy
> gain

## 7. §IV.D F1 — "quantization improved accuracy"

**Why:** a 4-bit model approximates the FP16 one; the generic-calibrated model
gained too; the gain is six scenarios on one judge pass; repeated passes over
identical FP16 answers disagreed. The baseline itself looks healthy: no
errors, timeouts, or empty answers. The domain corpus has no pump text, yet
pumps gained most. The regularization and LoRA explanations are speculation.

**Submitted:** the F1 paragraph from "F1. The combination of AWQ W4A16..."
through "...without the benefit of any gradient steps."

**Replace with:**
> F1. AWQ W4A16 with domain calibration lowers Qwen's mean latency 1.99×. Its
> judge pass rate was also higher than FP16 (81.8% vs 47.7%), but we treat
> this as provisional rather than as evidence that quantization improves
> accuracy. A 4-bit model approximates the FP16 one, the generic-calibrated
> model gained too (72.7%), and the gain consists of six scenarios that all
> flipped from fail to pass on a single judge pass (exact McNemar p = 0.031),
> while the two judge passes over the identical FP16 answers disagreed (12/22
> vs 9/22). The FP16 baseline shows no sign of being hobbled: no errors,
> timeouts, or empty answers. The domain corpus contains no pump text, yet
> pumps showed the largest gain (3/10 to 10/10), so the gain cannot be
> attributed to domain-matched calibration. A three-pass re-run and a
> human-labelled check will decide whether it holds.

## 8. §IV.D F2 — calibration comparison

**Add after the F2 numbers:**
> Because the domain corpus is far smaller and narrower than the generic one
> (Section III-C), these comparisons confound calibration topic with corpus
> size.

## 9. §IV.D F3 — the L2 collapse

**Submitted:**
> The most likely reason for this is that the FP8 KV-cache quantization
> corrupts the long visual-token prefix in VLMs (Vision-Language Models).
> While the flags that would help with this (i.e., L2_prefix_cache,
> L2_chunked_prefill, and L2_fp8_kv) are set up, none of them have been run
> through the experiment yet.

**Replace with:**
> We suspect, but have not yet isolated, FP8 KV-cache numerics. Under vLLM
> 0.19's V1 engine the FP16 baseline already ran with prefix caching and
> chunked prefill (both on by default), so FP8 KV is the bundle's only
> correctness-relevant change. The broken outputs were degenerate: pump and
> motor answers collapsed to the token "addCriterion", turbine answers stopped
> after one word, and transformer answers stayed partly coherent. Failures
> concentrate on the more uniform images (plain grayscale castings, thermal
> images, blades against sky), matching vLLM issue #41343, which documents
> FP8 KV-cache output corruption on Qwen2-VL and Qwen2.5-VL with default
> scaling. The bundle's 1,157 ms p50 is therefore not a real speedup: the
> model emitted only a few tokens (median answer length 17 characters). The
> follow-up sweep isolates the cause: FP8 alone, FP8 with dynamically computed
> KV scales, FP8 without prefix caching, the full bundle with dynamic scales,
> and an FP8-only Llama control.

## 10. §IV.D F4 — the 512 px result

**Submitted:** the F4 paragraph from "F4. Downscaling images produces no
benefit..." through "...unacceptable when target features are small".

**Replace with:**
> F4. A 512-pixel cap gave no speedup. Half the scenario images (pumps at
> 512 px, motors at 320 px) are already at or below the cap, so for those 11
> Qwen scenarios L3 sent byte-identical requests and received byte-identical
> answers. On the other 11 (transformers and turbines), Qwen2.5-VL, which uses
> native dynamic resolution rather than fixed tiles, received fewer image
> tokens, but the model wrote longer answers about the smaller images: the
> per-scenario latency change tracks the answer-length change (r = 0.93).
> Qwen's mean slowdown (9,920 vs 9,437 ms) is therefore an output-length
> effect, not an encoder property. Llama was effectively unchanged (9,278 vs
> 9,203 ms). Per-asset differences on pumps and motors (Table III) cannot
> reflect downscaling, since those inputs were unchanged: in the committed
> judge pass, the L0 and L3 pump answers are identical and both score 3/5, so
> the 3/10 vs 6/10 gap in Table III comes entirely from the second judge pass.
> The follow-up sweeps 256, 384, 512, 768, and 1024 px with logged
> image-token counts.

## 11. Table II caption

**Submitted:** "LLAMA L1G DENOMINATOR IS 40 DUE TO A LOGGING GAP."

**Replace with:** "Two Llama L1g scenarios returned empty answers, which the
original judge skipped (denominator 40); counted as failures, L1g scores 16/44."

**Why:** in the committed sweep, scenarios 6 and 20 returned empty answers
after about one second with no error. They were not a logging gap, and they
were not the ~122 s token-cap runaways that the README previously described.

## 12. Table III caption

**Add:** "L3 pump and motor inputs are identical to L0 (images at most 512 px);
differences in those columns reflect judge variance, not downscaling."

## 13. §V.C future work, item 1

**Submitted:**
> Isolation of the type of L2 flag used: L2_prefix_cache, L2_chunked_prefill,
> L2_fp8_kv run individually to determine which type of flag is causing the
> collapse of the Qwen response. The leading theory is that corruption from
> the quantization of the FP8 KV-cache has negatively impacted the integrity
> of the long visual token prefix.

**Replace with:**
> Isolating the L2 collapse. Because prefix caching and chunked prefill are
> already on in the L0 baseline under vLLM's V1 engine, the ablation varies
> the FP8 KV cache: FP8 alone, FP8 with dynamically computed KV scales
> (--calculate-kv-scales), FP8 without prefix caching, the full bundle with
> dynamic scales, and an FP8-only Llama control. The failure pattern matches
> vLLM issue #41343 (FP8 KV-cache corruption on Qwen2-VL/Qwen2.5-VL with
> default scaling).

## 14. §VI Conclusion

**Submitted:**
> AWQ W4A16 with domain calibration is the only variant that simultaneously
> improves latency and accuracy on Qwen (9,437 to 4,752 ms; 47.7% to 81.8%
> LLM-judge pass) and yields a 2.46× Llama speedup. The L2 vLLM serving bundle
> produced the largest cross-family failure mode (4.7× Qwen speedup) with
> total response collapse (0/44 judge pass)

**Replace with:**
> AWQ W4A16 with domain calibration nearly halves Qwen's latency (9,437 to
> 4,752 ms) with a provisional judge-accuracy gain (47.7% to 81.8%), and yields
> a 2.46× Llama speedup. The L2 vLLM serving bundle collapsed Qwen's output
> entirely (0/44 judge pass; its lower latency reflects the collapse, not a
> real speedup) while leaving Llama coherent

## 15. Appendix — "How we verified correctness"

**Why:** one of the two judge passes behind Tables II and III is not in the
repository, so the accuracy numbers cannot be traced to committed rows.

**Submitted:**
> Every reported number in Section III was produced by re-running the harness
> ourselves and is traceable to a CSV row under results/.

**Replace with:**
> Latency and VRAM numbers trace to results/hpml_metrics.csv. The judge scores
> in results/llm_judge.csv are one of the two judge passes behind Tables II and
> III; the second pass is not in the repository. The follow-up re-run commits
> every judge pass, and scripts/report_numbers.py regenerates every reported
> number from results/.
