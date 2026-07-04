# Final Project — html_book Reading List

Pipeline: `INPUT.avi → Stabilization → Background Subtraction → Matting → Tracking → OUTPUT.avi`
No deep learning. `opencv-contrib` allowed but caps grade at 70 pts.

book path: `video_processing_git/html_book/`

---

## Priority 1 — Background Subtraction → `binary.avi`, `extracted.avi`
**ch08_background_subtraction.html**
- [ ] Why frame differencing alone fails
- [ ] Why Gaussians?
- [ ] Per-pixel — not per-region
- [ ] Why "ω/σ" specifically?
- [ ] Online updates — the practical recipe
- [ ] Failure modes
- [ ] Per-pixel GMM — threshold the background (demo)

*(FAQ #10: KNN/MOG2 allowed but must explain how they work in report.)*

---

## Priority 2 — Stabilization → `stabilize.avi`
**ch01_motion_representation.html**
- [ ] 1.3 Affine Transformation
- [ ] Why "homogeneous" coordinates?
- [ ] 1.4 Homography
- [ ] What "up to scale" really means
- [ ] Why "8 unknowns" instead of 9?

**ch02_optical_flow.html**
- [ ] 2.2 Lucas Kanade
- [ ] What the Taylor approximation actually does
- [ ] When LK works and when it doesn't
- [ ] Pyramid trick (worth knowing)
- [ ] 2.5 Harris Corner Detector
- [ ] Harris response — live (demo)

**ch03_motion_applications.html**
- [ ] 3.2 Hyperlapse → Where RANSAC appears
- [ ] RANSAC vs Least Squares — line fitting with outliers (demo)
- [ ] Why "minimum sample size"?
- [ ] How many iterations K?

*(FAQ #1: black borders after warping are OK. Don't blur — hurts matting.)*

---

## Priority 3 — Matting → `matted.avi`, `alpha.avi`
**ch10_video_matting.html**
- [ ] Why "trimap" instead of just "mask"
- [ ] Where the scribbles are critical
- [ ] 10.1 Trimap Estimation
- [ ] KDE in plain words
- [ ] Trimap demo — drag the boundary band width
- [ ] 10.2 Trimap Refinement — Opacity map
- [ ] 10.3 Matting
- [ ] 10.4 KDE — Kernel Density Estimation

*(alpha in [0,1]; FAQ #5: if can't save [0,1], scale ×255 → uint8.)*

---

## Priority 4 — Tracking → `OUTPUT.avi`, `tracking.json`
**ch04_kalman_particle.html**
- [ ] 4.1 Tracking — three common motion assumptions
- [ ] 4.2 Kalman Filter
- [ ] The state vector — what to put in it
- [ ] Kalman 1-D — predict & update fuse gaussians (demo)
- [ ] 4.3 Particle Filter *(optional alternative)*
- [ ] Condensation — particles chasing a target (demo)

*(tracking.json: per frame → [ROW, COL, HEIGHT, WIDTH].)*

---

## Optional / polish
**ch11_retargeting_patchmatch.html**
- [ ] 11.5 Inpainting — only if filling stabilization black borders instead of leaving them

**ch09_video_colorization.html**
- [ ] Geodesic distance idea — sibling tool for label propagation near boundaries

## Skip (not used by this pipeline)
- ch05 Texture Synthesis
- ch06 Layer Representation
- ch07 Video Magnification
