# Video Processing — Interactive HTML Book

An 11-chapter interactive companion to the **Video Processing (0512-4263)** course at Tel Aviv University, School of Electrical Engineering — lectures by **Shai Avidan**, collected by **Yaniv Korobkin** (Spring 2025).

Each chapter shows the original lecture text alongside a plain-English explanation and a live interactive demo so the math actually moves.

## What's inside

- `index.html` — landing page with cards for every chapter
- `ch01` … `ch11` — one chapter per Contents section of the PDF
- `styles.css` — shared dark theme

## How to read a chapter

Each page is structured as repeating triples:

1. **Original text** (teal block) — quoted from the lecture notes, verbatim.
2. **Plain-English explanation** (blue block) — rewritten for a complete beginner.
3. **Interactive demo** (amber-headed card) — drag sliders, click buttons, watch the algorithm run.

Every chapter ends with a **Rule of Thumb** card and a **Summary** table.

## Topics

| # | Chapter | Demos |
|--|---------|-------|
| 1 | Motion Representation | transform playground, drag-corner homography, pinhole projection |
| 2 | Optical Flow | LK 1-D tangent, constraint lines, aperture problem, Harris heatmap, HS α slider |
| 3 | Applications of Motion Estimation | super-resolution fusion, RANSAC line fitting |
| 4 | Kalman & Particle Filters | Kalman with three Gaussians, Condensation particle tracker |
| 5 | Texture Synthesis | state-space trajectory + rendered texture |
| 6 | Layer Image Representation | K-means in 2-D |
| 7 | Video Magnification | Eulerian α-magnification |
| 8 | Background Subtraction | per-pixel GMM threshold |
| 9 | Video Colorization | click-to-place geodesic distance heatmap |
| 10 | Video Matting | trimap with adjustable boundary band |
| 11 | Re-targeting & PatchMatch | propagation-arrow convergence |

## Run it

Just open `index.html` in any modern browser. Everything works offline.

Equations render via [MathJax](https://www.mathjax.org/) loaded from CDN; demos are vanilla JS + Canvas.

## License

Generated for personal study. The lecture content belongs to its authors (Shai Avidan, Yaniv Korobkin).
