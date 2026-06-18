# NEXT_STEPS.md

Actionable plan for catching up and contributing. Companion to [`REPO_GUIDE.md`](REPO_GUIDE.md) (see §8 state-vs-proposal and §11 risks for the reasoning behind these). Tailored for David, joining late with limited time.

Effort scale: **S** ≈ a few hours · **M** ≈ 1–2 days · **L** ≈ multi-day / needs team alignment.

---

## 0. Before writing any code — align with the team (S, do first)

These three answers change everything downstream. Ask in the team channel:

1. **fVDB:** Was the custom region-growing chunking chosen *instead of* fVDB, or is fVDB-backed chunking still expected for the report? (Proposal's headline contribution; the code has zero fVDB usage.)
2. **Ownership:** Which areas are unowned? Luca has no commits; you have only scaffolding commits. Pick a lane nobody else is actively on (suggested: evaluation/scalability — currently nobody's).
3. **Canonical NKSR script:** Is [`scripts/testing_nksr_recon.py`](scripts/testing_nksr_recon.py) (byte-identical duplicate of `nksr_reconstruction.py`) safe to delete?

Don't start the L items until #1 is answered.

---

## 1. Quick wins — do these first (each S, ~half a day total)

Low-risk, immediately useful, and they make you productive in the codebase:

- [ ] **Fix `view_mesh.py` filename bug.** [`scripts/view_mesh.py:14`](scripts/view_mesh.py) reads `outputs/possoin_reconstruction.ply`; it should be `poisson_reconstruction.ply` (what [`scripts/poisson_meshing.py`](scripts/poisson_meshing.py) actually writes). One-line fix that unbreaks the comparison viewer.
- [ ] **Fix `run_vdbfusion.py` `NameError`.** [`scripts/run_vdbfusion.py:40`](scripts/run_vdbfusion.py) references undefined `start`; add `start = datetime.now()` at the top of `main()` (mirrors the other scripts).
- [ ] **Unify data paths.** `poisson_config.yaml`, `vdbfusion_config.yaml`, and the hardcoded path in `pcd_meshing.py` still point at per-user `/work/scratch/<name>/`. Switch them to the shared `/work/courses/3dv/team13/2026-03-09_16.19.44/` (already done for `nksr_config.yaml`). Unblocks every teammate running baselines.
- [ ] **Delete or clearly mark `testing_nksr_recon.py`** (pending answer to §0.3) and remove the orphan `configs/testing_config.yaml` — both are drift hazards for anyone reading the repo.
- [ ] **Commit `REPO_GUIDE.md` and this file** so the team benefits from the survey.

Doing these gives you 4–5 small, reviewable commits and real familiarity with the layout.

---

## 2. Highest-value contribution — Scalability study (RQ1) (M)

**Nobody owns this and it's a core research question that is currently completely unmeasured.** This is the best use of limited time.

- [ ] Add instrumentation around `run_nksr()` ([`scripts/nksr_reconstruction.py:264`](scripts/nksr_reconstruction.py)): per-unit point count, wall time, and `torch.cuda.max_memory_allocated()` (reset per unit). Log to a CSV.
- [ ] Add a scene-size sweep: reconstruct progressively larger spatial crops of the scan (e.g. 25% / 50% / 100% bounding box) and record total time + peak GPU memory + output mesh size.
- [ ] Produce the **runtime/memory vs. scene-size curve** the proposal promises. Save figures to a results dir and commit them (outside git-ignored `outputs/`).
- [ ] Also capture per-unit success/failure: `run_nksr` silently swallows OOM/RuntimeError and skips units, so a "successful" run can be missing regions with no warning. A summary line ("N units, M failed, X% points reconstructed") is cheap and important for quality interpretation.

This single deliverable directly answers Research Question 1 and produces report-ready figures.

---

## 3. Second-best — Multi-method comparison harness (M)

The proposal's main deliverable is comparative evaluation, but evaluation is currently per-mesh-only and manual.

- [ ] Write `scripts/compare_methods.py` that runs `evaluate_mesh` ([`src/quality_assessment.py:462`](src/quality_assessment.py)) over NKSR / VDBFusion / Poisson / pcd meshes against the source cloud and emits **one comparison table** (CSV + Markdown) and a combined figure.
- [ ] This is effectively the `evaluate_multiple_meshes` the module docstring already advertises but never implemented — coordinate with Jeffrey (QA owner) so you don't collide.
- [ ] Add a short methodological note: metrics use the source scan as pseudo-ground-truth, so they reward fidelity-to-input, not truth, and can't credit plausible hole-filling. The report needs this caveat.

---

## 4. If time permits (S–M each)

- [ ] **Add the BPA baseline** — proposal explicitly lists Ball Pivoting; ~30 lines with `o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting` + a `configs/bpa_config.yaml`. Completes the baseline set with low effort.
- [ ] **Wire `_surface_smoothness`** ([`src/quality_assessment.py:173`](src/quality_assessment.py), defined but unused) into `evaluate_mesh` — a cheap extra robustness signal.
- [ ] **Clean dead NKSR config keys** (`complex_boundary_inset_m`, the whole `subsampling.complex_oom_fallback_levels` block) so the config reflects what the code actually reads.
- [ ] **Sync README** — its VDBFusion section describes a `dataset_type: kitti/las` mode the current `vdbfusion_reconstruction.py` no longer has.

---

## 5. Larger / team-dependent (L — only after §0.1)

- [ ] **fVDB integration**, if the team confirms it's still in scope. High effort; would replace/augment the custom chunking with fVDB IndexGrid-backed chunking. Read the fVDB docs first; scope a spike before committing.
- [ ] **Document the NKSR/fVDB install** properly in the README — currently relies on an undocumented shared `.pth` hack into the team conda env; a newcomer cannot reproduce the environment from `requirements.txt` alone.

---

## Suggested order for *you* specifically

1. §0 (ask the 3 questions) — today.
2. §1 quick wins — first coding session, builds familiarity.
3. §2 scalability study — your main contribution; unowned, high-impact, report-critical.
4. §3 comparison harness *or* §4 BPA, depending on what the team still needs and Jeffrey's plans.
5. §5 only if explicitly agreed.

Rationale: §2 and §3 map directly onto the proposal's two research questions and its main deliverable, are currently unowned, and produce concrete report artifacts — the fastest way to go from "absent" to "visibly contributing."
