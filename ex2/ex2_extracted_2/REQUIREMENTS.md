# HW2: Optical Flow and Video Stabilization (0512.4263 Video Processing 2025/2026)

Full requirements extracted from `HW2_2026.pdf`. Reference doc so PDF need not be re-read.

## Environment
- conda python 3.11: `conda create -n vp python=3.11 -y` then `conda activate vp`
- `pip install -r requirements.txt`

## Files provided
- `HW2_2026.pdf`, `input.avi`, `river1.png`, `river2.png`
- `lucas_kanade.py` (implement functions here)
- `main_river.py`, `main_tau_video.py` (run these; DO NOT change except tuning constants)
- `requirements.txt`

---

## Section 1 — General Questions (answers go in PDF report)
Keep answers short. Sometimes >1 correct answer — explain. First line of PDF = your IDs.

Recall optical flow formula: `0 = I_t + I_x*u + I_y*v` (2 unknowns u,v, 1 equation per pixel).

1. **Constant brightness constraint?**
   - a. How does it help solve optical flow between two images?
   - b. Correct in real-world scenarios? (what happens with reflective objects?)
2. **Aperture problem** — what is it, what can we do about it?
3. **How did Lucas-Kanade solve optical flow?** What did they assume about movement of each pixel? (hint: movement of neighbourhood of each pixel.)
4. **Is LK assumption true around object boundaries? Why?**
5. **Propose general idea to correctly find optical flow on object boundaries**, given any input you desire (except true movement of each pixel). E.g. depth map / label image. State which inputs you assume + general idea of solution.

---

## Section 2 — Lucas-Kanade Optical Flow

### Part 2.1 — Optical Flow one Step
Compute velocity fields (u,v) between I1 and I2, back-project I2→I1 (result ≈ I1, little movement). Use LK algorithm with 2D-translation warp. Run `main_river.py`, implement in `lucas_kanade.py`.
- I1 = `river1.png`, I2 = `river2.png`
- Constants in head of `main_river.py` need finetune/edit.

#### Q1. Implement `build_pyramid(image, num_levels)`
- Image pyramid = list of downsampled versions.
- Start: list with copy of original image.
- Each level: convolve `PYRAMID_FILTER` with image from previous level, then decimate by picking every 2nd entry (indexing). Append filtered→decimated to end.
- List length = `num_levels + 1`.
- Hint: `signal.convolve2d` with `boundary='symm'`, `mode='same'`.
- **NOT allowed:** `cv2.PyrDown` or any cv2 method (different decimation).
- Inputs: `image` np.ndarray (h,w), `num_levels` int. Output: `pyramid` list of np.ndarray, length num_levels+1.

#### Q2. Implement `lucas_kanade_step(I1, I2, window_size)`
- Receives two images + window_size. Computes per-pixel shift in x and y.
- Outputs two maps (shape of input images). First = x-axis flow, second = y-axis.
- Inputs: I1 (h,w) time t, I2 (h,w) time t+1, window_size int (window = window_size × window_size).
- Output: `(du, dv)` tuple of two np.ndarray each (h,w). `dv` = flow in rows, `du` = flow in columns.
- Open file for further documentation.

#### Q3. Implement `warp_image(image, u, v)`
- Back-projects I2 using u,v. Result ≈ I1.
- Inputs: `image` (h,w) time t+1, `u` 2d array (cols flow), `v` 2d array (rows flow).
- Output: `image_warp` (h,w).
- **Must support u,v shapes != image shape.** Update u,v to image shape:
  1. `cv2.resize` u and v to image shape.
  2. Normalize shift values by factor = ratio between image dim and shift matrix dim (factor for u uses #cols in u; factor for v uses #rows in v).
- Warping via `scipy.interpolate.griddata`:
  1. Grid-points = flattened `meshgrid` of 0:w-1 and 0:h-1.
  2. Values = `image.flatten()`.
  3. Points to interpolate = flattened meshgrid matrices + v and u (add them).
  4. `fill_value = np.nan`.
- Finally fill nan holes with source image values. Hint: `np.isnan(image_warp)`.

#### Q4. Run `main_river.py` up to comment line `ONE STEP LUCAS KANADE ENDS HERE`
- **4.1** Put `river_results/0_river_one_LK_step_result.png` in PDF and explain result.
- **4.2** Look at `river_results/2_after_one_lk_step.gif` and `river_results/1_original.gif`. Answer in PDF: **Why is the result imperfect?**

#### Q5. Implement `lucas_kanade_optical_flow(I1, I2, window_size, max_iter, num_levels)`
- Calculates LK optical flow, max_iter iterations per pyramid level. Output `(u,v)` (h,w) each. v=rows, u=cols.
- Algorithm:
  1. Resize image to `[K*(2^(num_levels-1))] × [M*(2^(num_levels-1))]` where K=ceil(h/(2^(num_levels-1))), M=ceil(h/(2^(num_levels-1))). [sic — doc says h for both] Can use `cv2.resize`.
  2. Build pyramids for both images.
  3. Init u,v as zeros shape of I1.
  4. For every level (smallest → full size):
     - a. Warp I2 of that level by current u,v.
     - b. Repeat num_iterations: LK step with I1 decimated of current level and current I2_warp → new I2_warp.
     - c. For every level not the image's level: resize (cv2.resize) to next pyramid level resolution (bigger) and scale u,v accordingly.

#### Q6. Run entire `main_river.py`
- Put **all png outputs** in PDF. Explain all results.
- Explain why `river_results/3_after_full_lk.gif` looks better now.

---

## Section 3 — Video Stabilization
Run `main_tau_video.py`, process `input.avi`. Implement in `lucas_kanade.py`.

Method: for each frame compute LK optical flow vs previous frame, accumulate warps back to frame 1.
- Warp frame k+1 to k, add warp k→1, use combined warp to bring k+1 to frame-1 coordinate system.
- `H_4^0 = H_1^0 * H_2^1 * H_3^2 * H_4^3` (compose homographies/shifts).
- Frame 1 copied as-is to output.

3 output videos:
- `ID1_ID2_stabilized_video.avi`
- `ID1_ID2_faster_stabilized_video.avi`
- `ID1_ID2_fixed_borders_stabilized_video.avi`

### Part 3.1

#### Q7. Implement `lucas_kanade_video_stabilization(input_video_path, output_video_path, window_size, max_iter, num_levels)` → None
Writes stabilized video to output path. Steps:
1. Open VideoCapture of input, read params.
2. Create output VideoWriter same params. Use `fourcc = cv2.VideoWriter_fourcc(*'XVID')`.
3. Convert first frame to grayscale, write as-is to output.
4. Resize first frame as in Full-LK: `K*(2^(num_levels-1)) × M*(2^(num_levels-1))`, K=ceil(h/(2^(num_levels-1))), M=ceil(h/(2^(num_levels-1))).
5. Create u,v sized to image.
6. Loop frames (use tqdm):
   - a. Resize to shape in (4).
   - b. Feed to `lucas_kanade_optical_flow` with previous frame.
   - c. From u,v maps compute mean over valid region (exclude half-window borders each side).
   - d. Update u,v to those mean values inside valid region.
   - e. Add u,v shift from previous frame diff so frame t normalized back to first frame.
   - f. Save updated u,v for next frame (for step e next frame).
   - g. Warp current frame with u,v.
   - h. (optional debug: save each frame to dir; erase before submit.)
7. Gracefully close all VideoCapture, destroy all windows.

#### Q8. Run `main_tau_video.py` → `ID1_ID2_stabilized_video.avi`
- Explain result. What happened to MSE between frames?
- Border effects OK (handled later). Video should look stabilized (not still, but stable).
- MSE between frames should be ≥40% lower than original.
- Reference: original MSE = **60.23**, LK stabilized = **26.73**.

### Part 3.2 — Faster LK Implementation

#### Q9. Implement `faster_lucas_kanade_step(I1, I2, window_size)`
- Faster than `lucas_kanade_step`. Compute u,v only at interest points (corners) when pyramid resolution big enough. Small levels → normal LK step; high levels → corners only.
- Use own Harris (from ex1) or `cv2.cornerHarris`.
- Output `(du, dv)` (h,w) each. dv=rows, du=cols.
- Steps:
  1. If image small enough (you design threshold) → return `lucas_kanade_step` result.
  2. Else find corners in I2, compute u,v only for those pixels.
  3. Return u,v maps all zeros except corner pixels.

#### Q10. Implement `faster_lucas_kanade_optical_flow`
- Copy of `lucas_kanade_optical_flow` but call `faster_lucas_kanade_step` instead of `lucas_kanade_step`.

#### Q11. Implement `lucas_kanade_faster_video_stabilization`
- Copy of `lucas_kanade_video_stabilization` but call `faster_lucas_kanade_optical_flow`.
- `fourcc = cv2.VideoWriter_fourcc(*'XVID')`. Border effects OK.
- Runtime ≥30% lower. MSE ≥30% lower than original.
- Reference: original=60.23, LK=26.73, FASTER=**29.03**.

#### Q12. Implement `lucas_kanade_faster_video_stabilization_fix_effects(input_video_path, output_video_path, window_size, max_iter, num_levels, start_rows, start_cols, end_rows, end_cols)` → None
- Fixes border effects by cutting constant portion of image; uses `faster_lucas_kanade_optical_flow`.
- `fourcc = cv2.VideoWriter_fourcc(*'XVID')`.
- Params: start_rows=#lines cut from top, start_cols=#lines cut from bottom, end_rows=#cols cut from left, end_cols=#cols cut from right. [labels per PDF table, somewhat mislabeled]
- Reference: original=60.23, FASTER+BORDERS CUT=**24.39**.

---

## General Notes
- May add helper functions called from given functions.
- **Don't change main files** (`main_tau_video.py`, `main_river.py`) — only set window-size, num iterations, pyramid levels. Choose best values. Graders test functions from your main file.
- If creating each video takes >1 hour on Tochna computers → points decreased.

## Files to submit (zip = `vp2025_ex2_ID1_ID2.zip`)
- `ex2_ID1_ID2.pdf` — non-programming answers. **First line = ID2.**
- `ID1_ID2_stabilized_video.avi`
- `ID1_ID2_faster_stabilized_video.avi`
- `ID1_ID2_fixed_borders_stabilized_video.avi`
- `lucas_kanade.py`
- `main_river.py`
- `main_tau_video.py`

Compilation errors of any sort = zero for that question. Double-check before submit. Only one of each pair submits.

---

## River output filenames (from main_river.py)
- `river_results/0_river_one_LK_step_result.png` (Q4.1)
- `river_results/1_original.gif`, `river_results/2_after_one_lk_step.gif` (Q4.2)
- `river_results/river_full_LK_step_result.png` (Q6)
- `river_results/3_after_full_lk.gif` (Q6)
- stats JSON: `RIVER_{ID1}_{ID2}_mse_and_time_stats.json`
- Tuning constants in main_river.py: `WINDOW_SIZE_RIVER`, `MAX_ITER_RIVER`, `NUM_LEVELS_RIVER`
