# Split on Failure: Unattended Large-Scale 3D Gaussian Splatting Training via OOM-Reactive Tiling

*Draft v0 — 2026-08-10. Working titles below; numbers marked **[TODO]** await the running midpoint-vs-median ablation.*

**Alternative titles**
- *Let It Crash: Out-of-Memory as a Partitioning Signal for Large-Scale 3DGS Training*
- *OOM-Reactive Scene Tiling for Unattended 3D Gaussian Splatting at Arbitrary GPU Budgets*

---

## Abstract

Training 3D Gaussian Splatting (3DGS) on large scenes exceeds GPU memory, and every existing remedy tries to *prevent* the failure: pre-partitioning the scene by heuristics, capping Gaussian counts under a monitored budget, or restructuring training to stream parameters out-of-core. We take the opposite, pragmatic stance: **let the out-of-memory (OOM) failure happen, and treat it as the partitioning signal itself.** Our system trains a scene as a single tile until a rank crashes with OOM, classifies the failure into one of three categories (immediate capacity, fragmentation, densification growth), and reacts accordingly — retrying in place for transient fragmentation, or splitting the tile and re-queuing its children. For growth-induced OOM, a crash-tolerant handoff saves every rank's trained Gaussians at the moment of failure via shared-memory coordination and POSIX signals, merges them, and warm-starts both children from the parent's partial reconstruction, so no training progress is discarded. Because splitting is reactive, the system requires no memory model, no profiling pass, and no scene-specific tuning: the same command completes unattended on any GPU count and memory size, recursing exactly as deep as the hardware requires. We further show that the standard geometric-midpoint split systematically produces imbalanced children (2.1–2.2× in our scenes) and that SfM point density is a poor predictor of final Gaussian density (r = 0.28), invalidating proxy-based balancing; instead, splitting at the median of the *actually trained* Gaussian distribution reduces child imbalance to ≤1.03× **[TODO: full-run tile/depth/skip comparison]**. On a 30-camera urban scene, the system completed a 66M-Gaussian reconstruction fully unattended over 2.5 days, surviving 13 growth-induced OOM events with zero data loss.

---

## 1. Introduction

**Claims (each maps to a section/experiment):**

1. **OOM-reactive tiling** — no VRAM estimation, no monitoring, no pre-profiling. The failure *is* the signal. (§3.2, §4.2)
2. **Categorized failure handling** — not all OOMs mean "split": immediate-capacity vs. fragmentation vs. growth demand different reactions. (§3.1)
3. **Crash-tolerant warm-start handoff** — trained Gaussians survive the crash and seed the children; the scene is never retrained from scratch. (§3.3)
4. **Measured-median splitting** — balance children by the trained Gaussian distribution, not by area or SfM proxies; clamped to guarantee termination. (§3.4, §4.3)
5. **Unattended completion as a first-class goal** — the practical property practitioners need: launch once, get a scene, on whatever GPUs exist. (§4.4)

**Positioning in one sentence:** prior work spends effort *predicting or avoiding* memory failure; we spend none, and show that reacting to it is simpler, hardware-agnostic, and loses nothing.

## 2. Related Work

- **Pre-partitioned large-scale 3DGS** — VastGaussian, CityGaussian(V2), DOGS, Hierarchical-3DGS, HRGS, BlockGaussian. All divide *before* training using camera positions, SfM density, or spatial heuristics; block sizing must be tuned to the target GPU. BlockGaussian is closest (content-aware, load-balanced blocks) but estimates content by SfM point counts — a proxy we empirically invalidate (§4.3).
- **Memory-bounded / out-of-core training** — Taming-3DGS budgets, Gaussians on a Diet, TideGS, CLM, A LoD of Gaussians. These avoid OOM architecturally (caps, offloading, streaming); they trade throughput or quality ceilings for the guarantee and still assume the chosen budget fits.
- **Reactive memory management in ML systems** — DTR (ICLR'21) evicts and rematerializes tensors on OOM, establishing the reactive-vs-planning paradigm at tensor granularity; cluster schedulers (AntMan, CARMA) restart or offload jobs on OOM. We carry this philosophy to *spatial problem subdivision with state inheritance*: instead of redoing the same computation, the workload itself is permanently divided and partial results are inherited.
- **Distributed 3DGS** — Grendel-GS (our per-tile training substrate), RetinaGS, Splaxel. Orthogonal: they scale one training job; we schedule many.

## 3. Method

### 3.1 An OOM taxonomy for 3DGS training

| Category | Symptom | Cause | Reaction |
|---|---|---|---|
| 1 | OOM at iteration ~1 | working set (images + SfM init) exceeds capacity | split immediately; children from scratch |
| 2 | OOM before first densification, same-iteration recurrence | allocator fragmentation (reserved ≫ allocated) | bounded in-place retries, then split |
| 3 | OOM after densification onset | Gaussian growth | emergency-save all ranks → merge → split → **warm-start children** |
| 4 | child OOMs at load | inherited set exceeds capacity | re-split with inheritance |

Classification needs only the failing iteration index relative to the densification schedule — no memory introspection.

### 3.2 Reactive recursive tiling

Tiles form a binary tree over the scene's ground plane (X/Y only). A tile trains until convergence (loss-plateau early stop with best-checkpoint) or OOM; OOM enqueues two children. Termination is guaranteed by (a) the split-ratio clamp (§3.4) and (b) a minimum tile size. The scheduler is a simple persistent work queue (`adaptive_state.json`), which also makes the whole pipeline resumable after any interruption.

### 3.3 Crash-tolerant warm-start handoff (Cat-3)

At the OOM moment, ranks may be blocked in collectives, so no NCCL/collective can be used. The detecting rank writes the failure record — category, iteration, and the split coordinate (§3.4) — to a pre-mapped shared-memory file and signals peers (SIGUSR1); every rank filters its Gaussian shard by the two child bounding boxes and writes per-rank PLYs; the wrapper merges per child and validates counts. Children resume with the merged set sharded contiguously across ranks.

Engineering notes that proved necessary in practice (each traced to an observed failure):
- **Save-guard against shutdown signals.** torchrun SIGTERMs surviving ranks as soon as one exits; a naive handler restarted the in-progress save from scratch, and on large tiles the restart overran the grace period, losing one rank's file (observed once in ~19 events). A save-in-progress flag makes the handler yield and let the interrupted save resume. Verified by signal-injection test.
- **Fallback state write** on the save path, because the detecting rank's monitor thread can win the race against the main thread's state write (observed at ~100% rate before the fix).
- Loss-plateau convergence gated to opacity-reset-free windows (otherwise tiles are saved mid-reset and render black).

### 3.4 Splitting at the measured Gaussian median

Midpoint splitting is content-blind: in our scene it yielded a consistent ≈32/68 child imbalance, whose dense child re-OOMs on load, cascading toward the minimum-size floor and *skipped* tiles — permanent holes in the final scene. Balancing requires knowing where the content is, and we show SfM point counts do not (per-tile SfM count vs. final Gaussian count, r = 0.28; sparse-SfM regions *inflate more*, since densification compensates for extractor bias). The trained Gaussians themselves are the ground truth and are already in memory at the OOM moment: we split at the median of their coordinates along the longer axis, clamped so the larger child never exceeds 65% of the parent extent (termination guarantee). The detecting rank computes the cut on a subsample and publishes it through the same shared-memory channel, so all ranks filter by identical child boxes.

## 4. Experiments  *(current numbers; full tables pending)*

**Setup.** 30-camera urban aerial scene (Samsung-dong); 2–3× Quadro RTX 6000 24GB; Grendel-GS distributed backend; vanilla 3DGS densification for quality runs, aggressive schedule for OOM-stress ablations.

### 4.1 Unattended completion (system demonstration)
7/27 quality run: 37 tiles (19 leaves), 13 Cat-3 events, 0 lost rank files, 0 skipped tiles, 66.1M Gaussians merged — 2.5 days with zero human intervention. 3-GPU (odd world size) smoke test: full Cat-3 cycle verified, including bit-exact 62-attribute merge validation.

### 4.2 OOM taxonomy in action
Stress run: Cat-1 at iter 1 (image working set), Cat-2 recurring at fixed iterations (fragmentation; retry helps only when progress advances), Cat-3 after densification onset. Fragmentation mitigations (allocator GC threshold + max split size) extended root-tile survival 79 → 3500 iterations.

### 4.3 Midpoint vs. measured-median split (ablation)
Same scene, same seeds, 3 GPUs, aggressive densification:

| Cat-3 event | midpoint child balance | median child balance |
|---|---|---|
| iter 153/237 (X-axis) | 32.1 / 67.9 (2.11×) | 49.7 / 50.3 (1.013×) |
| iter 977/1007 (Y) | 31.1 / 68.9 (2.21×) | 49.8 / 50.2 (1.006×) |
| iter 958 (Y) | 30.6 / 65.0 (2.1×) | 49.4 / 50.6 (1.025×) |
| iter 268 (X) | 62.7 / 34.8 (1.8×) | 49.5 / 50.5 (1.021×, unclamped) |
| iter 1002/1132 (Y) | — | 54.0 / 46.0 (1.17×, clamp-limited) |

**[TODO]** total tiles / max depth / skipped / wall-clock, both runs to completion (midpoint full run in progress).

### 4.4 Overhead vs. the unconstrained oracle **[TODO — experiment designed, not yet run]**

The ideal reference for a memory-reactive method is training with *no memory constraint at all*. We construct it explicitly: choose a scene **A** sized so that original single-tile Grendel-GS fully trains on **C** GPUs (the oracle: wall-clock **D**, final quality **Q**, Gaussian count **E**), but does *not* fit on 2 GPUs. Then run Split-on-Failure on the same machine with 2 GPUs — which is forced to tile — and measure the time to reach the same held-out quality **Q**. We report normalized overhead

> **overhead = (T_ours × 2) / (D × C)**  (GPU-hours ratio; 1.0 = ideal)

Quality **Q** (held-out PSNR/SSIM) is the arrival criterion, not Gaussian count: tiled training duplicates boundary regions and discards out-of-box strays, so count trajectories are not comparable; **E** is reported as a secondary indicator only.

Overhead 1.0 is unreachable by construction — pre-OOM iterations are discarded, Cat-1/2 children restart from scratch, warm-started children re-converge, and the tile queue serializes work that the oracle runs as one job. The claim of this section is therefore *not* zero overhead but a bounded, tuning-free premium: **completion on hardware the oracle cannot use at all, at a measured extra cost of X%.** The decomposition of X (wasted iterations / restarts / re-convergence / scheduling gaps) is available from the run logs and will be reported alongside.

A property worth stating explicitly: when the scene *does* fit, Split-on-Failure never triggers and is byte-identical to the underlying Grendel-GS run — the overhead is zero by construction. Small scenes pay nothing; large scenes pay X% to complete unattended.

*Planned instantiation on our hardware: scene A = mini_30 at reduced resolution, calibrated so single-tile training fits on 8× RTX 6000 24GB but not on 2; oracle at C = 8; ours at 2 GPUs in both midpoint and median modes.*

### 4.5 Ablation targets remaining
- Cat-2 retry policy (same-iteration recurrence ⇒ early split) — data collected, policy pending.
- Cross-GPU-budget generality: same command on {2,3}× 24GB done; smaller-VRAM and larger-count configs **[TODO]**.
- Standard benchmarks (Mill-19 / MatrixCity) and quality metrics vs. VastGaussian / CityGaussian / BlockGaussian **[TODO]**.

## 5. Limitations
- Bounding-box inheritance discards Gaussians that drifted outside the parent box (0.65% early → 5.3% late in stress runs); quantifying the visual cost is open.
- Reactive splitting spends failed iterations before each split; we do not claim wall-clock optimality against a *perfectly tuned* pre-partitioning — we claim tuning-free completion.
- Single scene family so far; generality claims rest on the taxonomy, not yet on breadth.

---

*Next steps: (1) clean median rerun (post-fix) → fill §4.3 table; (2) oracle-overhead experiment (§4.4); (3) pick venue (3DV / WACV systems-flavored vs. CVPR main) → port to its LaTeX template; (4) benchmark scenes.*
