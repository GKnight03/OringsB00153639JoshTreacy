import os
import time
from collections import deque

import cv2
import numpy as np


def to_gray(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        return bgr.astype(np.uint8)
    b = bgr[:, :, 0].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)
    gray = 0.114 * b + 0.587 * g + 0.299 * r
    return np.clip(gray, 0, 255).astype(np.uint8)


def hist256(gray: np.ndarray) -> np.ndarray:
    h = np.zeros(256, dtype=np.int64)
    for v in gray.ravel():
        h[int(v)] += 1
    return h


def otsu_threshold(gray: np.ndarray) -> int:
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


def integral_image(bin01: np.ndarray) -> np.ndarray:
    ii = np.zeros((bin01.shape[0] + 1, bin01.shape[1] + 1), dtype=np.int32)
    ii[1:, 1:] = np.cumsum(np.cumsum(bin01.astype(np.int32), axis=0), axis=1)
    return ii


def window_sum(ii: np.ndarray, y0: int, x0: int, y1: int, x1: int) -> int:
    return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]


def erode(bin01: np.ndarray, k: int) -> np.ndarray:
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
    out = bin01.copy()
    for _ in range(iters):
        out = dilate(out, k)
        out = erode(out, k)
    return out


DIRS8 = [(-1, 0), (1, 0), (0, -1), (0, 1),
         (-1, -1), (-1, 1), (1, -1), (1, 1)]


def connected_components(bin01: np.ndarray):
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
    if not areas:
        return np.zeros_like(bin01), None
    idx = int(np.argmax(areas))
    lab = idx + 1
    return (labels == lab).astype(np.uint8), bboxes[idx]


def process_one(path: str, out_dir: str):
    img = cv2.imread(path)
    if img is None:
        return None

    t0 = time.perf_counter()

    gray = to_gray(img)
    t = otsu_threshold(gray)
    bin01 = (gray < t).astype(np.uint8)
    bin01 = close(bin01, k=5, iters=1)

    ring_mask, bbox = largest_component_mask(bin01)

    dt_ms = (time.perf_counter() - t0) * 1000.0

    base = os.path.splitext(os.path.basename(path))[0]
    cv2.imwrite(os.path.join(out_dir, f"{base}_ring_mask.png"), (ring_mask * 255).astype(np.uint8))

    out = img.copy()
    if bbox is not None:
        minx, miny, maxx, maxy = bbox
        cv2.rectangle(out, (minx, miny), (maxx, maxy), (0, 255, 0), 1)

    cv2.putText(out, f"Otsu t={t}", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(out, f"time={dt_ms:.2f}ms", (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(os.path.join(out_dir, f"{base}_annotated.png"), out)

    return {"file": os.path.basename(path), "otsu_t": t, "time_ms": dt_ms}


def main():
    in_dir = "."
    out_dir = "Orings_out"
    os.makedirs(out_dir, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [os.path.join(in_dir, f) for f in os.listdir(in_dir)
             if os.path.splitext(f.lower())[1] in exts]
    files.sort()

    for p in files:
        r = process_one(p, out_dir)
        if r:
            print(r["file"], "-> ring extracted", f"({r['time_ms']:.2f} ms)")

    print("Done. Outputs in:", out_dir)


if __name__ == "__main__":
    main()