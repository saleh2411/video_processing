"""Background subtraction: per-pixel adaptive GMM (course book ch. 8,
Stauffer-Grimson style) in CIE-Lab colour space, vectorized numpy.

Colour space: CIE-Lab separates illumination (L) from chroma (a, b).
Shadows, reflections and exposure wobble live almost entirely in L, while
the person's clothes/skin differ in chroma. The match distance therefore
down-weights L:

    d2 = 0.25*dL^2 + da^2 + db^2

so brightness events barely count and chroma changes get a full vote.
(Measured: this removes the "poster corner glued to the head" artifact that
a BGR distance produces when the person walking by darkens a glossy poster.)

Model: each pixel keeps K Gaussians (mu_k in Lab, scalar sigma_k, weight
w_k). Gaussian 0 is initialized from the temporal MEDIAN colour and the
MAD (the person covers any pixel for a minority of frames, so the median
IS the background), with a wider sigma floor at high-gradient pixels
(edges/text flicker with any sub-pixel stabilization jitter). Training
then runs the chapter's streaming updates: 2.5-sigma match test, weight
grow/decay + renormalize, rho-nudged mu/sigma, replace-weakest on no match.

Classification: Gaussians sorted by w/sigma; top ones holding cumulative
weight T_BG = 0.7 form the background set, the lecture's own suggested
value. A higher T risks letting the person's own "ghost" mode become
background where he lingers, but this was not observed on this video.
Cast shadows are vetoed exactly: L attenuated into [0.50, 0.96] while
|da|, |db| < 8. Warp borders excluded via a validity mask from the model.

Post-processing: open -> close -> keep components >= 20% of the largest
(limbs may split at dark clothing) -> re-close -> fill holes -> LINEAR
upscale + threshold + median filter + one erode for a smooth full-res
boundary.

Outputs: extracted video (colour person on black), binary video
(person = 255 per spec FAQ #5, else 0).
"""
from tqdm import tqdm
import utils
import cv2
import numpy as np


K = 4                  # Gaussians per pixel (chapter: 3-5)
ALPHA = 0.01           # weight learning rate
T_BG = 0.7             # background cumulative-weight threshold (top mode only)
MATCH_SIGMA = 2.5      # match test width
SIGMA_INIT = 20.0      # sigma for NEW (replacement) Gaussians
SIGMA_MIN, SIGMA_MAX = 3.0, 10.0   # cap stops person inflating a bg Gaussian
W_NEW = 0.05           # weight of a replacement Gaussian
PROC_SCALE = 0.5       # GMM runs at this scale; masks upscaled at the end

L_WEIGHT = 0.25        # weight of the L channel in the match distance
EFF_CH = L_WEIGHT + 2.0            # effective channel count for thresholds
CH_W = np.array([L_WEIGHT, 1.0, 1.0], np.float32)

# cast shadow in Lab: L attenuated, chroma (a, b) nearly unchanged
SHADOW_L_LO, SHADOW_L_HI = 0.50, 0.96
SHADOW_CHROMA_MAX = 8.0

# mask post-processing kernels (at processing scale)
OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))


def train_gmm(small_frames):
    """Pass 1: online GMM updates over the whole video. Returns mu, sigma, w."""
    h, w_ = small_frames[0].shape[:2]
    mu = np.zeros((K, h, w_, 3), np.float32)
    sigma = np.full((K, h, w_), SIGMA_INIT, np.float32)
    weight = np.zeros((K, h, w_), np.float32)

    # offline initialization: temporal median = the true background colour,
    # MAD = its true noise width; gradient-adaptive sigma floor tolerates
    # jitter flicker at edges/text
    stack = np.stack(small_frames)                       # [N,H,W,3]
    med = np.median(stack, axis=0)
    mad = np.median(np.abs(stack - med[None]), axis=0).mean(axis=2)
    gray_med = med[..., 0]                               # L of the median
    gmag = (np.abs(cv2.Sobel(gray_med, cv2.CV_32F, 1, 0, ksize=3)) +
            np.abs(cv2.Sobel(gray_med, cv2.CV_32F, 0, 1, ksize=3)))
    sigma_floor = np.clip(0.15 * gmag, SIGMA_MIN, SIGMA_MAX)
    mu[0] = med
    sigma[0] = np.clip(np.maximum(1.4826 * mad, sigma_floor),
                       SIGMA_MIN, SIGMA_MAX)
    weight[0] = 1.0
    del stack

    ks = np.arange(K)[:, None, None]
    for frame in tqdm(small_frames[1:], desc="BG model (train)"):
        diff = frame[None] - mu                       # [K,H,W,3]
        dist2 = np.einsum('khwc,khwc,c->khw', diff, diff, CH_W)
        match = dist2 < (MATCH_SIGMA * sigma) ** 2 * EFF_CH

        score = np.where(match, weight / sigma, -1.0)
        k_best = score.argmax(axis=0)                 # [H,W]
        matched_any = match.any(axis=0)
        upd = (k_best[None] == ks) & matched_any[None]  # one-hot winner

        # weights: winner grows, everyone else decays
        weight = (1 - ALPHA) * weight + ALPHA * upd

        # mu / sigma of the winner move toward the evidence
        rho = np.minimum(ALPHA / np.maximum(weight, 1e-4), 0.25)
        rho_u = np.where(upd, rho, 0.0)[..., None]
        mu = (1 - rho_u) * mu + rho_u * frame[None]
        sig2 = sigma ** 2
        sig2 = np.where(upd, (1 - rho) * sig2 + rho * dist2 / EFF_CH, sig2)
        sigma = np.sqrt(np.clip(sig2, SIGMA_MIN ** 2, SIGMA_MAX ** 2))

        # no match anywhere -> replace this pixel's weakest Gaussian
        repl_px = ~matched_any                        # [H,W]
        if repl_px.any():
            k_weak = weight.argmin(axis=0)
            repl = (k_weak[None] == ks) & repl_px[None]
            mu = np.where(repl[..., None], frame[None], mu)
            sigma = np.where(repl, SIGMA_INIT, sigma)
            weight = np.where(repl, W_NEW, weight)

        weight /= weight.sum(axis=0, keepdims=True)

    return mu, sigma, weight


def background_set(sigma, weight):
    """Chapter's decision rule: sort by w/sigma desc, background = top
    Gaussians until cumulative weight exceeds T_BG. Returns bool [K,H,W]."""
    order = np.argsort(-(weight / sigma), axis=0)     # [K,H,W]
    w_sorted = np.take_along_axis(weight, order, axis=0)
    cum = np.cumsum(w_sorted, axis=0)
    in_bg_sorted = (cum - w_sorted) < T_BG            # first b with cum > T
    is_bg = np.empty_like(in_bg_sorted)
    np.put_along_axis(is_bg, order, in_bg_sorted, axis=0)
    return is_bg


def classify_frame(frame, mu, sigma, is_bg, bg_mean):
    """Foreground mask for one (downscaled, Lab) frame. uint8 {0,1}."""
    diff = frame[None] - mu
    dist2 = np.einsum('khwc,khwc,c->khw', diff, diff, CH_W)
    match = dist2 < (MATCH_SIGMA * sigma) ** 2 * EFF_CH
    background = (match & is_bg).any(axis=0)

    # cast shadow: L attenuated, chroma nearly unchanged
    l_ratio = frame[..., 0] / np.maximum(bg_mean[..., 0], 1.0)
    chroma_close = ((np.abs(frame[..., 1] - bg_mean[..., 1]) < SHADOW_CHROMA_MAX) &
                    (np.abs(frame[..., 2] - bg_mean[..., 2]) < SHADOW_CHROMA_MAX))
    shadow = (l_ratio > SHADOW_L_LO) & (l_ratio < SHADOW_L_HI) & chroma_close
    background |= shadow

    return (~background).astype(np.uint8)


def clean_mask(mask):
    """Morphology + connected-component filtering + hole filling."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, OPEN_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, CLOSE_KERNEL)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_area = areas.max()
        keep = 1 + np.where(areas >= 0.2 * largest_area)[0]
        mask = np.isin(labels, keep).astype(np.uint8)
        # re-fuse the kept fragments (legs/torso split by dark clothing)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, CLOSE_KERNEL)

    # fill internal holes: anything not reachable from the border is person
    inv = (1 - mask).astype(np.uint8)
    n_i, labels_i = cv2.connectedComponents(inv, connectivity=4)
    border_labels = np.unique(np.concatenate([
        labels_i[0, :], labels_i[-1, :], labels_i[:, 0], labels_i[:, -1]]))
    holes = inv.astype(bool) & ~np.isin(labels_i, border_labels)
    mask[holes] = 1
    return mask


def background_subtraction(input_video_path, extracted_video_path,
                           binary_video_path):
    """Extract the walking person from the stabilized video."""
    # read once: full-res BGR frames for output, half-res Lab for the GMM
    cap = utils.open_video(input_video_path)
    params = utils.get_video_parameters(cap)
    w_full, h = params["width"], params["height"]
    fps = params["fps"]

    frames, small_frames = [], []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        small = cv2.resize(frame, None, fx=PROC_SCALE, fy=PROC_SCALE,
                           interpolation=cv2.INTER_AREA)
        small_frames.append(
            cv2.cvtColor(small, cv2.COLOR_BGR2Lab).astype(np.float32))
    cap.release()

    mu, sigma, weight = train_gmm(small_frames)
    is_bg = background_set(sigma, weight)
    top_k = weight.argmax(axis=0)
    bg_mean = np.take_along_axis(
        mu, top_k[None, ..., None], axis=0)[0]        # [H,W,3] dominant colour

    # warp-border black pixels (black has L ~ 0) are never person
    valid = (bg_mean[..., 0] > 15).astype(np.uint8)
    valid = cv2.erode(valid, np.ones((7, 7), np.uint8))

    out_extracted = utils.create_video(extracted_video_path, fps, w_full, h)
    out_binary = utils.create_video(binary_video_path, fps, w_full, h)

    for frame, small in tqdm(list(zip(frames, small_frames)),
                             desc="BG model (classify)"):
        mask = classify_frame(small, mu, sigma, is_bg, bg_mean)
        mask &= valid
        mask = clean_mask(mask)

        # smooth full-res boundary: linear upscale + threshold kills the
        # half-res staircase; median filter rounds it organically; one erode
        # compensates the close-kernel bloat so edges hug the person
        mask_full = cv2.resize(mask.astype(np.float32), (w_full, h),
                               interpolation=cv2.INTER_LINEAR)
        mask_full = ((mask_full > 0.5) * 255).astype(np.uint8)
        mask_full = cv2.medianBlur(mask_full, 9)
        mask_full = cv2.erode(mask_full, np.ones((3, 3), np.uint8))
        mask_full = (mask_full > 127).astype(np.uint8)

        extracted = frame * mask_full[..., None]
        binary = (mask_full * 255).astype(np.uint8)

        out_extracted.write(extracted.astype(np.uint8))
        out_binary.write(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))

    out_extracted.release()
    out_binary.release()
    cv2.destroyAllWindows()