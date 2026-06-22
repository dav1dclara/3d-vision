# REPO_GUIDE.md

Methodological guide to the ETH Zurich 3D Vision project **"LiDAR to Mesh using Neural Kernel Surface Reconstruction"**. The focus here is on *why the pipeline is built the way it is* — the design decisions, their rationale, and their trade-offs — not on code structure. Implementation specifics (line numbers, function names) are deliberately omitted.

---

## 1. Problem framing

The goal is to turn a building-scale indoor **NavVis VLX** LiDAR scan (millions of points, uneven density, occlusions, no ground-truth mesh) into a single colored triangle mesh, using **Neural Kernel Surface Reconstruction (NKSR)** as the core mesher. The two research questions are **(RQ1) scalability** of NKSR to whole-building scans and **(RQ2) reconstruction quality/robustness** versus classical baselines.

NKSR is a learned implicit-surface method: from oriented points it predicts compactly-supported kernels on a sparse voxel hierarchy, solves a sparse linear system for an implicit field, and extracts a dual mesh. It is accurate but its memory and runtime scale with the number of points and the reconstructed volume — so a whole building **cannot** be reconstructed in a single call on one GPU. Everything interesting in this repo is the strategy that makes NKSR usable at building scale while keeping quality high. That strategy is the subject of §3–§5.

> **Largest divergence from the proposal:** the proposal's headline contribution was integrating NKSR with **fVDB** sparse voxel grids for chunked processing. The current code uses **no fVDB at all** — the chunking is a custom geometry-aware scheme described below. Whether fVDB is still in scope is the single most important open question (§8).

---

## 2. Why the obvious approaches don't work

Two naive ways to scale NKSR to a building, and why they were rejected — this motivates the whole design:

- **Single global reconstruction.** Out of memory. NKSR's cost grows with point count and volume; a full scan exceeds GPU memory.
- **Regular-grid chunking** (split the scene into uniform boxes, reconstruct each, stitch). Two failure modes:
  1. **Seams fall in the worst places.** A regular grid cuts arbitrarily through surfaces and, critically, through *object–wall transitions* (a chair where it meets the floor). Reconstructing the two sides independently produces mismatched surfaces and visible stitching artifacts exactly where geometry is most delicate.
  2. **Uniform resolution is wasteful.** A flat wall and an intricate chair are reconstructed at the same (high) resolution. Most of an indoor scene is flat — so most of the compute is spent over-resolving surfaces that a coarse mesh would capture perfectly.

The pipeline is engineered specifically to avoid both. The central idea: **decompose the scene by geometric difficulty rather than by a fixed grid, so that (a) hard regions are reconstructed together with their surrounding context, and (b) the seams that remain fall on flat, easy surfaces where they can be trimmed cleanly.**

---

## 3. The main pipeline — methodological choices

The pipeline is: classify local geometry → group into reconstruction units by difficulty → reconstruct each unit adaptively → merge. Each step below is a deliberate methodological decision, given with its rationale and trade-off.

### 3.1 Local geometry classification: planar vs. complex
The scene is first partitioned into fine voxel cells (0.25 m), and each cell is classified as **planar** or **complex** by fitting a plane with SVD and taking the ratio of the smallest to largest singular value as a normalized **planarity residual** (≈0 means the points lie in a plane).

- **Why classify at all:** indoor scenes are dominated by large flat surfaces (walls, floors, ceilings) that are *easy* to reconstruct, punctuated by *hard* complex geometry (furniture, objects, people). Treating these two classes differently is what enables both the quality and the efficiency wins downstream.
- **Why a singular-value ratio:** it is scale- and density-normalized, so a single global threshold (`residual_threshold`, default 0.1) behaves consistently across the scene regardless of how many points a cell holds.
- **Why 0.25 m cells:** small enough to localize where geometry is complex, large enough to contain a statistically meaningful plane fit (cells below `min_points_per_chunk`, default 20, are ignored as noise).
- **Trade-off:** the threshold is global and hand-tuned. Cells near the boundary (textured walls, gentle curves) are sensitive to it; too low over-classifies as complex (slow), too high misses real detail. This is the dominant quality/runtime lever and a natural place for the RQ1/RQ2 sensitivity study.

### 3.2 Complex-first region growing with planar absorption
This is the **core methodological insight** of the project. Complex cells grow into reconstruction units first (breadth-first), and as a complex region grows it **greedily absorbs every adjacent cell, including flat ones**.

- **Why complex regions absorb their flat surroundings:** the hardest part of reconstruction is the *transition* between an object and the surface it rests on. If a complex region carries the surrounding wall/floor into the *same* NKSR call, the network sees the full context of every edge and produces a clean, continuous transition — no stitching. This is precisely the seam-placement problem from §2, solved by construction.
- **Why complex-first (priority):** complex geometry dictates where the difficult seams are, so it claims territory first; flat surfaces are subordinate context.
- **Bounded growth for memory:** complex-to-complex growth is capped by spatial extent (`complex_max_extent_m`, default 2 m) and every unit is capped by point count (`complex_max_pts`, default 1 M). These caps are the explicit knob that keeps each unit within GPU memory — the scalability mechanism. Note the asymmetry: only complex-to-complex expansion counts against the extent cap; absorbed flat context can extend beyond it, because context is cheap and valuable.
- **Trade-off / risk:** when a large complex region hits the point/extent cap it stops, and the remaining complex cells form a *separate* unit. Because complex units are deliberately **not** trimmed (see §3.4), two adjacent complex units can then overlap at their shared boundary (double surfaces / z-fighting). The design trades a possible seam *between complex units* (rare, on genuinely large objects) for guaranteed-clean object–wall transitions (common). Worth measuring how often the cap actually splits a region.

### 3.3 Leftover flat surfaces: coplanar planar units
Flat cells never absorbed by any complex region are grown into their own **planar-only** units. Here growth is gated by **two** predicates: normals must be similar (`angle_threshold_deg`, 15°) **and** the cells must be genuinely coplanar (mean point-to-plane distance below `coplanar_dist_threshold`, 0.1 m).

- **Why both predicates:** two parallel walls have nearly identical normals but different plane offsets. Normal-similarity alone would merge them into one unit; the coplanarity (offset) test prevents that, so a unit corresponds to a single physical surface.
- **Why separate units for flat regions:** they can be reconstructed far more cheaply than complex ones (§3.4), and keeping them apart from complex units is what lets the system spend its compute budget where it matters.

### 3.4 Adaptive resolution per unit class
Each unit is reconstructed by NKSR at a resolution matched to its difficulty:
- **Complex units:** highest detail, finest voxel — full fidelity for furniture/objects.
- **Planar units, tiered:** "very flat" surfaces (residual below `planar_very_flat_threshold`) use the coarsest voxel and lowest detail; merely "flat" surfaces use an intermediate setting.

- **Why adaptive:** flat surfaces reconstruct correctly at low resolution, so resolving them finely buys nothing but GPU time. Spending the resolution budget only where curvature/detail exists is the efficiency half of the §2 motivation. The tiering even within planar reflects that a pristine wall can go coarser than a slightly textured one.
- **Trade-off:** the per-class parameters are hand-set, not derived. They encode an assumption that "flat ⇒ low frequency" which holds for walls/floors but could under-resolve flat-but-detailed surfaces (e.g. a flat panel with fine relief).

### 3.5 Asymmetric overlap-and-trim at unit boundaries
How adjacent units are kept consistent differs by class, on purpose:
- **Planar units** are reconstructed on an **expanded** point set (core bounding box plus an overlap margin, `planar_overlap_m`, 0.3 m), then the resulting mesh is **trimmed back to the core box**. The overlap gives NKSR enough context that the implicit field doesn't fall off at the unit edge; trimming to the core ensures neighbouring planar units meet without double-covering.
- **Complex units** are **never trimmed** — they are meant to *win* every transition zone they cover.

- **Why asymmetric:** it operationalizes the priority from §3.2. Complex geometry owns transitions; flat surfaces yield to it and only tile amongst themselves. The trim is deliberately a simple core-box clip of the overlap — an earlier design that subtracted complex bounding boxes from planar meshes was abandoned because it could delete whole regions.
- **Trade-off:** the trim box is axis-aligned. At planar boundaries that are not axis-aligned, a box clip can leave small gaps or slivers of overlap. And the no-trim complex policy is what creates the complex–complex overlap risk noted in §3.2.

### 3.6 Subsampling: only where it's free
Planar units (after overlap expansion) can be huge, so they are voxel-downsampled until under a point cap (`planar_max_pts`, 300 k). Complex units are **not** subsampled — their size is controlled by region growth instead.

- **Why:** downsampling a flat surface loses no reconstructable signal (the plane is over-determined), so it's an essentially free memory saving. Downsampling complex geometry *would* lose detail, so detail there is preserved and bounded the other way (smaller regions).

### 3.7 Trusting scanner normals
Normals are read directly from the NavVis-exported PLY rather than estimated from the points.

- **Why:** every implicit method here (NKSR and Poisson) depends critically on *consistent, oriented* normals; estimating and globally orienting normals on a building-scale cloud is slow and error-prone. The scanner already provides them.
- **Trade-off / risk:** the whole pipeline inherits the quality of those normals. If they are noisy or inconsistently oriented in places, NKSR degrades there with no safeguard — a worthwhile thing to sanity-check.

### 3.8 Color as a decoupled texture field
Geometry is reconstructed from xyz + normals; color is attached afterward as a learned NKSR texture field sampled from the colored input points. Geometry and appearance are kept independent — color never influences the surface.

### 3.9 Robustness: per-unit failure isolation
If NKSR fails on a unit (out-of-memory or a fatal CUDA error), that unit is skipped rather than crashing the run, and the reconstructor is reinitialized to recover from a corrupted CUDA context.

- **Why:** a multi-hour building-scale run should not be lost to one pathological unit.
- **Trade-off (important for evaluation):** failures are silent. A "successful" run can be missing whole regions with no summary of what was dropped — so quality metrics could look fine while coverage is incomplete. A per-unit success/coverage report is the obvious missing safeguard and matters directly for RQ2.

### 3.10 Streaming the final merge
Per-unit meshes are streamed into one binary PLY a single chunk at a time, so the complete building mesh never has to reside in memory at once. This is a scalability decision, not a cosmetic one — it removes the final whole-scene memory bottleneck (and supports a `--merge-only` re-merge without re-running the GPU work).

---

## 4. The design in one paragraph

Decompose the scene by **difficulty, not geometry-blind grid**. Classify local cells as flat or complex; let complex regions grow first and **absorb their flat surroundings** so every object–wall transition is reconstructed with full context in one NKSR call; let the remaining flat surfaces form cheap coplanar units. Reconstruct each unit at a resolution matched to its difficulty, make **complex units win all transitions (no trim)** while flat units **overlap-and-trim** against each other, subsample only where it costs no quality, and stream-merge so nothing is bounded by total scene size. The result places unavoidable seams on flat, forgiving surfaces and spends compute only where geometry demands it.

---

## 5. Where this could be challenged (methodological open questions)

These are the points a reviewer (or the report) should probe:

1. **Global planarity threshold.** One number governs the planar/complex split for the entire scene; results are sensitive to it and it is untuned per scene. No sensitivity analysis exists.
2. **Order-dependent, greedy decomposition.** Region growing iterates cells in an arbitrary order; a different order can yield a different partition into units, hence a different mesh. Determinism/robustness of the decomposition is unquantified.
3. **Complex–complex seams from the memory cap.** The point/extent caps can split a large object into untrimmed adjacent units that overlap. Frequency and visual impact unmeasured.
4. **Axis-aligned trimming** of planar overlaps can misbehave on non-axis-aligned surface boundaries.
5. **Silent unit failures** (§3.9) can produce incomplete meshes that still score well — a validity threat for RQ2.
6. **Evaluation has no ground truth.** Quality is measured against the *source scan itself* as a pseudo-ground-truth, so metrics reward fidelity-to-input (including its noise) and cannot credit plausible hole-filling. This is acknowledged in the proposal but needs an explicit methodological caveat and ideally a held-out or cross-method agreement check.
7. **fVDB absence** (§1) changes the report's framing of the contribution and must be resolved with the team.

---

## 6. Baselines — chosen to span reconstruction paradigms

The baselines are not arbitrary; each represents a different surface-reconstruction philosophy, which is the point of comparing against them:

- **VDBFusion — volumetric TSDF + Marching Cubes.** Fuses a truncated signed-distance volume, then polygonizes. The classical robust baseline; the repo includes a parameter sweep over voxel size / truncation / weight.
- **Screened Poisson — global implicit.** Solves a Poisson equation for an indicator function from oriented points; needs good normals (which the project has); density trimming removes hallucinated exterior surface. The closest classical analogue to NKSR's implicit-field approach.
- **Ball Pivoting (BPA) — interpolating/local.** Rolls a ball over the points to triangulate them directly (no implicit field). Mirrors a validated MeshLab pipeline (resample → normals → orient → pivot). Reconstructs only where data exists — no hole-filling — which makes it a useful contrast to the implicit methods.
- **pcdmeshing — block Delaunay / visibility.** A tetrahedralization-plus-visibility mesher; an extra comparison point beyond the proposal's named baselines.

Comparing a learned implicit method (NKSR) against volumetric (TSDF), global-implicit (Poisson), interpolating (BPA), and Delaunay/visibility (pcdmeshing) is what lets RQ2 say something general about *why* NKSR wins or loses, not just *that* it does.

> Methodological caveat carried by the baseline configs: they currently point at a **mix of scenes** (NKSR uses the newer large-scale scan; Poisson/pcd reference the older small scene) and per-user paths. Cross-method numbers are only comparable on a common input — unify the input scene before drawing RQ2 conclusions.

---

## 7. Evaluation methodology (quality assessment)

A dedicated tool compares a reconstructed mesh against a reference point cloud and reports cloud-to-mesh distance statistics (Hausdorff / RMSE / MAE), a residual distribution binned into Good/OK/Critical/Missing, mesh structure and watertightness/manifoldness, a bidirectional F-score, and a residual heatmap. It is built to scale (memory-mapped random sampling of the cloud; Open3D's C++ ray-casting BVH for distance queries), so it handles very large clouds.

The methodologically important points (beyond §5.6):
- The reference is a **point cloud, not a mesh** — fidelity is measured against samples of the input scan.
- Evaluation is **per-mesh only**: there is no automated multi-method comparison harness, so cross-method comparison (the heart of RQ2) is currently manual. Building that harness is the most direct way to turn the existing tooling into a report deliverable.

---

## 8. State vs. proposal

| Item | Status |
|---|---|
| NKSR core + geometry-aware chunking | ✅ Done, mature |
| Adaptive resolution + transition-aware seam placement | ✅ Done (the contribution that exists) |
| VDBFusion / Poisson / BPA / pcdmeshing baselines | ✅ All implemented |
| Quality metrics + GUI (scales to large clouds) | ✅ Done |
| **fVDB integration** (proposal headline) | ❌ Absent — resolve scope with team |
| **Multi-method comparison harness** | ❌ Missing — RQ2 deliverable |
| **RQ1 scalability study** (runtime/memory vs. scene size) | ❌ Not captured — only wall-clock prints |
| Per-unit coverage/failure reporting | ❌ Missing — validity risk for RQ2 |

The two unmeasured research questions (RQ1 scalability curves, RQ2 multi-method comparison) are the highest-value gaps; both are about *measurement and methodology*, not new pipeline code.

---

## 9. Repository cruft (brief)

Minor hazards that confuse newcomers but don't affect the methodology: a diverged stale copy of the NKSR script (`scripts/testing_nksr_recon.py`), an orphan `configs/testing_config.yaml`, dead keys in `nksr_config.yaml` (an old OOM-fallback block), a `NameError` in `run_vdbfusion.py`, a hardcoded/typo'd `view_mesh.py`, and README drift (stale cell size, output name, and a removed VDBFusion `dataset_type` mode).

---

## 10. Environment & running

A conda env from `requirements.txt` **plus** a shared site-packages `.pth` into the team's `nksr` conda env supplies the heavy GPU deps (`torch`, `nksr`, `vdbfusion`, … are not in `requirements.txt`). A CUDA GPU is required.

```bash
conda activate 3DV
python scripts/reconstruction/nksr_reconstruction.py               # main pipeline → outputs/nksr_reconstruction.ply
python scripts/reconstruction/run_ball_pivoting.py                 # BPA baseline
python scripts/reconstruction/vdbfusion_reconstruction.py --config configs/vdbfusion_config.yaml
python scripts/reconstruction/poisson_meshing.py
python scripts/reconstruction/pcd_meshing.py
python scripts/run_quality_assessment.py                           # Tkinter GUI; needs a display
```
The NKSR script also exposes `--save-boundaries` / `--save-voxel-grid` (export the decomposition as wireframes — useful for explaining the method in the report) and `--merge-only` (re-merge existing chunks without GPU work).
