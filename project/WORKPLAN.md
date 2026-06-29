# Final Project — Parallel Work Plan (2 people)

Video Processing 0512-4263. Pipeline: **Stabilization → Background Subtraction → Matting → Tracking**.
Goal: two people work in parallel without blocking each other, on a pipeline that is naturally sequential.

The trick: **freeze the function interfaces on day 1**, give each person separate files, and let each develop against *mock inputs* until real ones are ready.

---

## 1. The split (balanced)

| | **Person A — front half + infra** | **Person B — back half + tracking** |
|---|---|---|
| Owns modules | `video_utils.py`, `stabilization.py`, `background_subtraction.py` | `config.py`, `matting.py`, `tracking.py` |
| Near drop-in (reuse) | **Stabilization** ← ex2 `lucas_kanade.py` | **Tracking** ← ex3 `particle_filter.py` |
| Build fresh (hard) | **Background Subtraction** | **Matting** (hardest module) |
| Also owns | shared video I/O helpers | constants / paths config |
| Report sections | Stabilization, Background Subtraction | Matting, Tracking |
| JSON | `timing.json` (with main) | `tracking.json` |

**Why balanced:** each person gets exactly one "port-and-adapt" module (low new code) and one "build-from-scratch" module (high effort). B's fresh module (matting, ★★★★★) is the single hardest, so B carries less infra; A owns `video_utils.py` to compensate.

**Why it parallelizes:** A produces the binary mask that B's matting consumes — but B mocks a mask until A's is ready, so neither waits.

---

## 2. Code structure

```
FinalProject/
├── Code/
│   ├── main.py                  # orchestrator: load once, call stages, write outputs, stamp timing.json   [shared — Phase 0]
│   ├── config.py                # IDs, paths, per-stage tunable constants                                   [B]
│   ├── video_utils.py           # load_frames / save_frames / save_binary / get_video_parameters            [A, built Phase 0]
│   ├── stabilization.py         # stabilize()                                                               [A]
│   ├── background_subtraction.py# subtract()                                                                [A]
│   ├── matting.py               # matte()                                                                   [B]
│   └── tracking.py              # track()                                                                   [B]
├── Input/      (INPUT.avi, background.jpg)
├── Outputs/    (6 avi + timing.json + tracking.json)   ← created by main.py
├── Temp/       (optional scratch / mock masks)
└── Document/   (report PDF)
```

**One module = one stage = one owner.** Separate files → zero merge conflicts. The only shared-edit files are `main.py`, `config.py`, `video_utils.py` — those are frozen in Phase 0 and touched together after that.

---

## 3. Frozen interface contracts  ⚠️ agree these on day 1, then do not change without telling the other person

All stages pass **NumPy frames in memory** (BGR `uint8`), not file paths — avoids re-reading videos between stages (helps the 20-min budget). `main.py` does all disk writes so timing stamps are accurate.

```python
# video_utils.py  (A)
def load_frames(path) -> tuple[list[np.ndarray], dict]:
    """Return (frames as BGR uint8 list, params dict{fourcc,fps,height,width,frame_count})."""

def save_frames(path, frames, fps) -> None:
    """Write color frames to .avi (XVID)."""

def save_binary(path, masks, fps) -> None:
    """masks are 0/1 (HxW). Scale ×255, uint8, write as video."""

# stabilization.py  (A)
def stabilize(frames, params) -> list[np.ndarray]:
    """In: raw frames. Out: stabilized color frames (same count, same size)."""

# background_subtraction.py  (A)
def subtract(stab_frames, params) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Out: (extracted_frames  -> person pixels else 0, color),
            (binary_masks      -> 0/1 single-channel, person=1)."""

# matting.py  (B)
def matte(stab_frames, binary_masks, bg_image, params) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """In: stabilized frames + 0/1 masks + new background (BGR).
       Out: (matted_frames color), (alpha_frames in [0,1] float HxW)."""

# tracking.py  (B)
def track(matted_frames, params) -> tuple[list[np.ndarray], dict]:
    """Out: (output_frames with rectangle drawn),
            (boxes dict {frame_index(int): [ROW, COL, HEIGHT, WIDTH]})."""
```

**The mask contract is the critical handshake:** `binary_masks[i]` is `HxW`, values in `{0,1}`, `1 = person`. A produces it, B consumes it. Lock this exact shape/dtype before splitting.

---

## 4. Phases & checkpoints

### Phase 0 — Foundation (TOGETHER, ~half a day)
- [ ] Agree the contracts in §3 verbatim.
- [ ] A writes `video_utils.py` (load/save/binary). Test: load INPUT.avi → save copy → plays.
- [ ] B writes `config.py` (IDs, relative paths via `os.path`, empty constant blocks per stage).
- [ ] Write `main.py` with **stub stages** that just pass frames through unchanged.
- [ ] **Checkpoint 0:** `python Code/main.py` runs end-to-end and writes 6 (dummy) videos + both JSON files into `Outputs/`. Pipeline plumbing proven before any real algorithm exists.
- [ ] Freeze interfaces. Split.

### Phase 1 — Parallel build (each works alone against mocks)
Person A and Person B work independently — see §5. Sync only at the daily check-in.
- [ ] **Checkpoint A1:** stabilize() produces a visibly steadier video.
- [ ] **Checkpoint A2:** subtract() produces a clean-ish binary mask + extracted video.
- [ ] **Checkpoint B1:** matte() composites a mock-masked person onto background.jpg.
- [ ] **Checkpoint B2:** track() draws a box following the person + fills tracking.json.

### Phase 2 — Integration (TOGETHER, ~half a day)
- [ ] Replace B's mock mask with A's real `subtract()` output.
- [ ] Run full real pipeline INPUT → OUTPUT. Fix shape/dtype/colour mismatches.
- [ ] **Checkpoint 2:** all 6 real videos correct + both JSONs valid (run the JSON test snippet from the brief).

### Phase 3 — Tune + report (TOGETHER + split writing)
- [ ] Quality pass: clean masks (morphology), sharpen matting edges, tighten tracking box.
- [ ] Runtime pass: keep total **< 20 min**. Vectorize hot loops, downscale where safe, `tqdm` to find the slow stage.
- [ ] Robustness pass: test that constants aren't overfit (different clothes / background in secret video).
- [ ] Each writes their own report sections; assemble PDF together.
- [ ] Record the screen-capture run for `ScreenRec/`.
- [ ] Final folder/naming check (case-sensitive, no spaces, IDs substituted, nothing extra in `Outputs/`).

---

## 5. How each person works without waiting

### Person A
- **stabilization.py** — needs only INPUT.avi. Zero dependency, start immediately. Port ex2 `lucas_kanade_faster_video_stabilization_fix_effects`; adapt to **color** (warp all 3 channels with the same u,v), strip file-I/O (work on in-memory frames per contract).
- **background_subtraction.py** — needs stabilized frames. Mock = use raw INPUT frames (or A's own early stabilize output). Build fresh: background model (KNN/MOG2 or color-space stats) → threshold → morphology (open kills speckle, close fills holes) → keep largest connected component.

### Person B
- **matting.py** — needs stab frames + binary mask. Mock both: raw INPUT as "stab", and a quick throwaway mask (threshold or one hand-made mask) to develop against. Build fresh: trimap from mask → alpha estimate in the boundary band → `out = α·fg + (1−α)·bg`.
- **tracking.py** — needs matted video. Mock = run on raw INPUT (or the ex3 sample frames) to validate the particle-filter port first, then swap to real matted. Port ex3 `particle_filter.py`, **strip the matplotlib `show_particles` display**, convert particle state → `[ROW,COL,HEIGHT,WIDTH]`.

A throwaway `Temp/make_mock_mask.py` that A writes early (even a rough threshold mask) lets B start matting on something realistic on day 1.

---

## 6. Adaptation gotchas (carry over from the ex1–3 reuse review)

| Module | Gotcha |
|---|---|
| stabilization | ex2 code is grayscale → must warp **color** frames; `crop_resize` blur can hurt matting, tune trim margins |
| tracking | strip matplotlib plotting (brief forbids on-screen figures); state→box conversion |
| matting / bg-sub | both fresh — no prior code; binary mask shape/dtype must match the §3 contract exactly |
| all | color output required everywhere except `binary.avi`; relative paths only; ends gracefully, no user input |

---

## 7. Git workflow

- `main` branch stays runnable.
- Branch per module: `a/stabilization`, `a/bgsub`, `b/matting`, `b/tracking`.
- Separate files = clean merges. Edit `main.py` / `config.py` / `video_utils.py` **only in Phase 0** (or in a quick joint session if an interface must change — then both pull immediately).
- Daily: each pushes their branch + a one-line status at the check-in (done / blocked / interface question).

---

## 8. Report (PDF) split

Each person documents the stages they built (the brief grades the report heavily and requires explaining any non-class algorithm or "too-specific" function with source):
- Intro + flow diagram + results table → together.
- Stabilization, Background Subtraction → A.
- Matting, Tracking → B.
- Runtime/quality discussion + benchmark → together.

---

### One-line summary
A owns **stabilize + background-subtract + video I/O**; B owns **matting + tracking + config**. Freeze the contracts in §3 on day 1, prove the empty pipeline runs (Checkpoint 0), then build in parallel against mocks and integrate. Balanced: each gets one easy port + one hard fresh module.
