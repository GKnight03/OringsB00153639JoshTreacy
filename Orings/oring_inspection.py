import os
import time
import math
from collections import deque

import cv2
import numpy as np


# ----------------------------
# 1) Basic helpers (no cv2 ops except imread/imwrite; implement grayscale conversion, histogram, Otsu thresholding from scratch)
# ----------------------------

def to_gray(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR -> grayscale (uint8) without cv2.cvtColor."""
    if bgr.ndim == 2:
        return bgr.astype(np.uint8)
    b = bgr[:, :, 0].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)
    gray = 0.114 * b + 0.587 * g + 0.299 * r
    return np.clip(gray, 0, 255).astype(np.uint8)


def hist256(gray: np.ndarray) -> np.ndarray:
    """Compute 256-bin histogram without cv2."""
    h = np.zeros(256, dtype=np.int64)
    flat = gray.ravel()
    for v in flat:
        h[int(v)] += 1
    return h


def otsu_threshold(gray: np.ndarray) -> int:
    """
    Otsu threshold selection from histogram.
    Returns threshold t such that foreground = gray < t (dark object on light bg).
    """
    h = hist256(gray)
    total = gray.size

    sum_total = 0.0
    for i in range(256):
        sum_total += i * h[i]

    sum_b = 0.0
    w_b = 0
    var_max = -1.0
    best_t = 0

    for t in range(256):
        w_b += h[t]
        if w_b == 0:
            sum_b += t * h[t]
            continue

        w_f = total - w_b
        if w_f == 0:
            break

        sum_b += t * h[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f

        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > var_max:
            var_max = var_between
            best_t = t

    return int(best_t)


# -----------------------------------------
# 2) Fast binary morphology (integral image for fast box sum queries, then threshold for erosion/dilation); no cv2 morphology ops
# -----------------------------------------

def integral_image(bin01: np.ndarray) -> np.ndarray:
    """Integral image for 2D array (uint8/0-1)."""
    # pad with a zero row/col to make area queries easy
    ii = np.zeros((bin01.shape[0] + 1, bin01.shape[1] + 1), dtype=np.int32)
    ii[1:, 1:] = np.cumsum(np.cumsum(bin01.astype(np.int32), axis=0), axis=1)
    return ii


def window_sum(ii: np.ndarray, y0: int, x0: int, y1: int, x1: int) -> int:
    """
    Sum over [y0,y1) x [x0,x1) using integral image ii.
    ii is (H+1, W+1).
    """
    return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]


def erode(bin01: np.ndarray, k: int) -> np.ndarray:
    """Binary erosion with square kxk, implemented via integral image."""
    assert k % 2 == 1
    r = k // 2
    H, W = bin01.shape
    padded = np.pad(bin01, ((r, r), (r, r)), mode="constant", constant_values=0)
    ii = integral_image(padded)

    out = np.zeros((H, W), dtype=np.uint8)
    area = k * k

    for y in range(H):
        for x in range(W):
            s = window_sum(ii, y, x, y + k, x + k)
            out[y, x] = 1 if s == area else 0
    return out


def dilate(bin01: np.ndarray, k: int) -> np.ndarray:
    """Binary dilation with square kxk, implemented via integral image."""
    assert k % 2 == 1
    r = k // 2
    H, W = bin01.shape
    padded = np.pad(bin01, ((r, r), (r, r)), mode="constant", constant_values=0)
    ii = integral_image(padded)

    out = np.zeros((H, W), dtype=np.uint8)

    for y in range(H):
        for x in range(W):
            s = window_sum(ii, y, x, y + k, x + k)
            out[y, x] = 1 if s > 0 else 0
    return out


def close(bin01: np.ndarray, k: int, iters: int = 1) -> np.ndarray:
    """Closing = dilation then erosion."""
    out = bin01.copy()
    for _ in range(iters):
        out = dilate(out, k)
        out = erode(out, k)
    return out


# --------------------------------------
# 3) Connected component labelling (CCL) by BFS; also get areas and bboxes for each CC
# --------------------------------------

DIRS8 = [(-1, 0), (1, 0), (0, -1), (0, 1),
         (-1, -1), (-1, 1), (1, -1), (1, 1)]


def connected_components(bin01: np.ndarray):
    """
    8-connected component labelling by BFS.
    Returns labels (int32), areas list, bboxes list (minx,miny,maxx,maxy).
    """
    H, W = bin01.shape
    labels = np.zeros((H, W), dtype=np.int32)

    current = 0
    areas = []
    bboxes = []

    for y in range(H):
        for x in range(W):
            if bin01[y, x] == 1 and labels[y, x] == 0:
                current += 1
                q = deque([(y, x)])
                labels[y, x] = current

                area = 0
                miny = maxy = y
                minx = maxx = x

                while q:
                    cy, cx = q.popleft()
                    area += 1
                    if cy < miny: miny = cy
                    if cy > maxy: maxy = cy
                    if cx < minx: minx = cx
                    if cx > maxx: maxx = cx

                    for dy, dx in DIRS8:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W:
                            if bin01[ny, nx] == 1 and labels[ny, nx] == 0:
                                labels[ny, nx] = current
                                q.append((ny, nx))

                areas.append(area)
                bboxes.append((minx, miny, maxx, maxy))

    return labels, areas, bboxes


def largest_component_mask(bin01: np.ndarray):
    labels, areas, bboxes = connected_components(bin01)
    if len(areas) == 0:
        return np.zeros_like(bin01), None
    idx = int(np.argmax(areas))
    lab = idx + 1
    mask = (labels == lab).astype(np.uint8)
    return mask, bboxes[idx]


# ------------------------------------------
# 4) "Fill holes" for centroid (not for ring defects) estimation (flood-fill from border, then any remaining 0s are holes -> set to 1)
# ------------------------------------------

def fill_holes(bin01: np.ndarray) -> np.ndarray:
    """
    Flood-fill background from border; any remaining 0s are holes -> set to 1.
    This will fill the donut's inner hole as well (useful to estimate true center).
    """
    H, W = bin01.shape
    visited = np.zeros((H, W), dtype=np.uint8)
    q = deque()

    # enqueue border background
    for x in range(W):
        if bin01[0, x] == 0: q.append((0, x)); visited[0, x] = 1
        if bin01[H - 1, x] == 0 and not visited[H - 1, x]:
            q.append((H - 1, x)); visited[H - 1, x] = 1

    for y in range(H):
        if bin01[y, 0] == 0 and not visited[y, 0]:
            q.append((y, 0)); visited[y, 0] = 1
        if bin01[y, W - 1] == 0 and not visited[y, W - 1]:
            q.append((y, W - 1)); visited[y, W - 1] = 1

    dirs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while q:
        y, x = q.popleft()
        for dy, dx in dirs4:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                if visited[ny, nx] == 0 and bin01[ny, nx] == 0:
                    visited[ny, nx] = 1
                    q.append((ny, nx))

    out = bin01.copy()
    holes = (bin01 == 0) & (visited == 0)
    out[holes] = 1
    return out


# -----------------------------------------
# 5) Defect analysis by radial/polar sampling from estimated center
# -----------------------------------------

def radial_features(ring_mask: np.ndarray, n_angles: int = 720):
    """
    From a (largest) ring mask, estimate:
    - missing angles (no ring found)
    - fraction of angles that see >1 foreground segment (broken/open ring symptom)
    - thickness stats and inner/outer radius variation
    """
    # Estimate geometric center using filled disk centroid (fills inner hole, but not outer gaps/breaks, which is what we want for a stable center).
    filled = fill_holes(ring_mask)
    ys, xs = np.nonzero(filled)
    if len(xs) == 0:
        return None

    cy = float(np.mean(ys))
    cx = float(np.mean(xs))

    H, W = ring_mask.shape
    maxR = math.hypot(max(cx, W - 1 - cx), max(cy, H - 1 - cy))

    segcount = np.zeros(n_angles, dtype=np.int32)
    inner = []
    outer = []
    thick = []

    for i in range(n_angles):
        ang = 2 * math.pi * i / n_angles
        dy = math.sin(ang)
        dx = math.cos(ang)

        steps = int(maxR) + 1
        vals = []

        for s in range(steps):
            y = int(round(cy + dy * s))
            x = int(round(cx + dx * s))
            if y < 0 or y >= H or x < 0 or x >= W:
                break
            vals.append(ring_mask[y, x])

        # count number of foreground segments along this ray (1 segment = normal ring; >1 segments = broken/open ring symptom; 0 segments = missing sector or segmentation failure)
        c = 0
        in_fg = False
        for v in vals:
            if v == 1 and not in_fg:
                c += 1
                in_fg = True
            elif v == 0 and in_fg:
                in_fg = False

        segcount[i] = c

        if c == 0:
            continue

        # take the FIRST segment (closest ring material along that direction); if multiple segments, it's a broken/open ring symptom (we see the far side of the break)
        s0 = None
        s1 = None
        for idx, v in enumerate(vals):
            if v == 1 and s0 is None:
                s0 = idx
            if v == 0 and s0 is not None and s1 is None:
                s1 = idx - 1
                break
        if s1 is None:
            s1 = len(vals) - 1

        inner.append(s0)
        outer.append(s1)
        thick.append(s1 - s0 + 1)

    seg_gt1_frac = float(np.mean(segcount > 1))
    missing_frac = float(np.mean(segcount == 0))

    if len(thick) == 0:
        return None

    inner = np.array(inner, dtype=np.float32)
    outer = np.array(outer, dtype=np.float32)
    thick = np.array(thick, dtype=np.float32)

    return {
        "cx": cx, "cy": cy,
        "missing_frac": missing_frac,
        "seg_gt1_frac": seg_gt1_frac,
        "inner_mean": float(np.mean(inner)),
        "inner_std": float(np.std(inner)),
        "outer_mean": float(np.mean(outer)),
        "outer_std": float(np.std(outer)),
        "thick_mean": float(np.mean(thick)),
        "thick_std": float(np.std(thick)),
        "thick_min": float(np.min(thick)),
    }


def classify_orings(feat: dict):
    """
    Heuristic classifier (tune thresholds if needed after you inspect outputs).
    """
    if feat is None:
        return "FAIL", "No ring detected"

    # broken/open ring often causes rays to hit multiple segments (you see the far side of the break); tune threshold if needed after inspecting outputs (we want to allow small cracks but not large breaks)
    if feat["seg_gt1_frac"] > 0.01:
        return "FAIL", "Broken/open ring"

    # if rays see no ring at all (large missing sector or segmentation failure), it's a fail; tune threshold if needed after inspecting outputs (we want to allow small gaps but not large ones)
    if feat["missing_frac"] > 0.01:
        return "FAIL", "Missing sector"

    # thickness checks: if min thickness is very low, it's likely a thin/missing material defect; if std is high, it's likely a non-uniformity/flash/bump defect
    if feat["thick_min"] < 0.55 * feat["thick_mean"]:
        return "FAIL", "Thin/missing material"

    if feat["thick_std"] > 0.20 * feat["thick_mean"]:
        return "FAIL", "Thickness variation"

    # radius variation checks (bumps/flash/shape deformation); tune if needed
    if feat["outer_std"] > 0.12 * feat["outer_mean"]:
        return "FAIL", "Outer radius variation"

    if feat["inner_std"] > 0.12 * feat["inner_mean"]:
        return "FAIL", "Inner radius variation"

    return "PASS", "OK"


# -----------------------
# 6) Full pipeline per image + main loop over directory
# -----------------------

def process_one(path: str, out_dir: str):
    img = cv2.imread(path)
    if img is None:
        return None

    t0 = time.perf_counter()

    gray = to_gray(img)

    # automatic threshold (Otsu) to separate dark ring from light background; no need for manual tuning here
    t = otsu_threshold(gray)

    # object is darker than background => foreground = gray < t
    bin01 = (gray < t).astype(np.uint8)

    # morphology: close small cracks/holes in the rubber (do NOT "repair" huge breaks that indicate real defects)
    # tune k/iters if needed after inspecting outputs; we want to close small gaps but not large ones that indicate real defects
    bin01 = close(bin01, k=5, iters=1)

    # largest CC is the O-ring (hopefully); also get its bbox for visualization
    ring, bbox = largest_component_mask(bin01)

    # features + classify
    feat = radial_features(ring, n_angles=720)
    verdict, reason = classify_orings(feat)

    dt_ms = (time.perf_counter() - t0) * 1000.0

    # annotate output image with bbox, verdict, reason, timing, and save a visualization of the mask as well
    out = img.copy()
    if bbox is not None:
        minx, miny, maxx, maxy = bbox
        cv2.rectangle(out, (minx, miny), (maxx, maxy), (0, 255, 0), 1)

    cv2.putText(out, f"Threshold (Otsu): {t}", (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(out, f"Result: {verdict} ({reason})", (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(out, f"Proc time: {dt_ms:.2f} ms", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # save a visual of the mask as well (largest CC in white, rest black)
    mask_vis = (ring * 255).astype(np.uint8)
    mask_vis = (ring * 255).astype(np.uint8)
    mask_vis = np.stack([mask_vis, mask_vis, mask_vis], axis=2)  # Gray->BGR without cv2
    cv2.putText(mask_vis, "Largest CC mask", (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, f"{base}_annotated.png")
    mask_path = os.path.join(out_dir, f"{base}_mask.png")

    cv2.imwrite(out_path, out)
    cv2.imwrite(mask_path, mask_vis)

    return {
        "file": os.path.basename(path),
        "otsu_t": t,
        "verdict": verdict,
        "reason": reason,
        "time_ms": dt_ms,
        **({} if feat is None else {
            "missing_frac": feat["missing_frac"],
            "seg_gt1_frac": feat["seg_gt1_frac"],
            "thick_mean": feat["thick_mean"],
            "thick_std": feat["thick_std"],
            "thick_min": feat["thick_min"],
        })
    }


def main():
    
    in_dir = "."         
    out_dir = "Orings_out"    # outputs here
    os.makedirs(out_dir, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [os.path.join(in_dir, f) for f in os.listdir(in_dir)
             if os.path.splitext(f.lower())[1] in exts]
    files.sort()

    results = []
    for p in files:
        r = process_one(p, out_dir)
        if r is not None:
            results.append(r)
            print(r["file"], "->", r["verdict"], "-", r["reason"], f"({r['time_ms']:.2f} ms)")

    # save a simple CSV summary
    csv_path = os.path.join(out_dir, "summary.csv")
    if results:
        keys = sorted(results[0].keys())
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for r in results:
                f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

    print("Done. Outputs in:", out_dir)


if __name__ == "__main__":
    main()