#!/usr/bin/env python3
"""
Dipendenze:  pip install rosbags numpy opencv-python
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# --------------------------------------------------------------------------- #
# Conversione messaggi
# --------------------------------------------------------------------------- #


def image_msg_to_numpy(msg):
    """sensor_msgs/Image -> ndarray (gestisce step e encoding comuni)."""
    enc = msg.encoding
    h, w = msg.height, msg.width
    data = np.asarray(msg.data, dtype=np.uint8)

    if enc in ("32FC1", "32FC"):
        itemsize, dtype, ch = 4, np.float32, 1
    elif enc in ("16UC1", "mono16"):
        itemsize, dtype, ch = 2, np.uint16, 1
    elif enc in ("mono8", "8UC1"):
        itemsize, dtype, ch = 1, np.uint8, 1
    elif enc in ("bgr8", "rgb8"):
        itemsize, dtype, ch = 1, np.uint8, 3
    elif enc in ("bgra8", "rgba8"):
        itemsize, dtype, ch = 1, np.uint8, 4
    else:
        raise ValueError(f"Encoding non supportato: {enc}")

    rows = data.reshape(h, msg.step)[:, : w * ch * itemsize]
    arr = np.ascontiguousarray(rows).view(dtype).reshape(h, w, ch)
    arr = arr[:, :, 0] if ch == 1 else arr
    return arr, enc


def to_gray_float(arr, enc):
    if arr.ndim == 2:
        g = arr
    elif enc == "bgra8":
        g = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
    elif enc == "rgba8":
        g = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
    elif enc == "rgb8":
        g = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:  # bgr8
        g = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return g.astype(np.float32)


def stamp_ns(msg):
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)

ST_OK = 0          # accettato
ST_TEXTURE = 1     # varianza locale (left) sotto --min-texture
ST_NOCAND = 2      # nessun candidato valido (bordo sinistro o finestra right piatta)
ST_SCORE = 3       # miglior score NCC sotto --min-score
ST_MARGIN = 4      # miglior match non abbastanza distinto dal secondo (--min-margin)
ST_LR = 5          # fallito il controllo left-right (--lr-thresh)

STATUS_NAMES = {
    ST_TEXTURE: "low texture",
    ST_NOCAND: "no candidate",
    ST_SCORE: "low NCC score",
    ST_MARGIN: "low margin",
    ST_LR: "L-R check fail",
}


# --------------------------------------------------------------------------- #
# NCC stereo matching (immagine di riferimento = left)
# Algorithm used is ZNCC (Zero mean Normalized Cross Correlation)
# ZNCC is more robust to difference in exposure of the images,
#   as well as increasing the absoulte distance between wrong patches 
# Complexity for each run O(H x W x max_disparity)
# Params:
#   max_disp: the maximum disparity checked (128 uses min range of zed2i)
#   min_disp: the min disparity checked (0 is infinite distance)
#   min_texture: the min variance for each window, (low variance means textureless surface --> wrong distance estimation)
#   subpixel: bool, perform subpixel refinement
#   
# --------------------------------------------------------------------------- #


def ncc_disparity(L, R, max_disp=128, win=20, min_disp=0, min_score=0.85, min_texture=5.0, subpixel=True,
                  min_margin=0.0, lr_thresh=None):
    """
    Block matching con Zero-mean Normalized Cross-Correlation.
    Il pixel (y, x) in L corrisponde a (y, x - d) in R.
    Ritorna (disparity float32, score NCC float32). Pixel non validi = NaN.
    """
    H, W = L.shape
    k = (win, win)
    # Average the image with a win x win filter of all 1
    box = lambda a: cv2.boxFilter(a, cv2.CV_32F, k, borderType=cv2.BORDER_REPLICATE) #pad with border pixel

    # Mean and variance for both left and right images
    mL = box(L)
    vL = np.maximum(box(L * L) - mL * mL, 0)
    mR = box(R)
    vR = np.maximum(box(R * R) - mR * mR, 0)

    n_d = max_disp - min_disp + 1
    volume = np.full((n_d, H, W), -1.0, np.float32)

    #main zncc algo
    for i, d in enumerate(range(min_disp, max_disp + 1)):
        if d >= W:
            break
        Rs = np.zeros_like(R)
        Rs[:, d:] = R[:, : W - d]
        mRs = np.zeros_like(mR)
        mRs[:, d:] = mR[:, : W - d]
        vRs = np.zeros_like(vR)
        vRs[:, d:] = vR[:, : W - d]

        cov = box(L * Rs) - mL * mRs
        ncc = cov / np.sqrt(vL * vRs + 1e-6)
        ncc[(vL < min_texture) | (vRs < min_texture)] = -1.0
        ncc[:, :d] = -1.0
        volume[i] = ncc

    best_i = np.argmax(volume, axis=0)
    best_s = np.take_along_axis(volume, best_i[None], 0)[0]
    disp = (best_i + min_disp).astype(np.float32)

    # Subpixel value rifinement (change disparity by at most half by checking neighbours)
    if subpixel:
        im = np.clip(best_i - 1, 0, n_d - 1)
        ip = np.clip(best_i + 1, 0, n_d - 1)
        sm = np.take_along_axis(volume, im[None], 0)[0]
        sp = np.take_along_axis(volume, ip[None], 0)[0]
        den = sm - 2 * best_s + sp
        ok = (best_i > 0) & (best_i < n_d - 1) & (np.abs(den) > 1e-6)
        off = np.zeros_like(disp)
        off[ok] = 0.5 * (sm[ok] - sp[ok]) / den[ok]
        disp += np.clip(off, -0.5, 0.5)

    disp_int = best_i + min_disp

    m_tex = vL < min_texture
    m_nocand = (best_s <= -1.0) & ~m_tex
    m_score = (best_s < min_score) & ~m_tex & ~m_nocand
    m_margin = np.zeros((H, W), bool)
    m_lr = np.zeros((H, W), bool)

    # 1) unicita': il miglior match deve staccarsi dal secondo miglior candidato
    if min_margin > 0:
        idx = np.arange(n_d)[:, None, None]
        second = np.where(np.abs(idx - best_i[None]) <= 1, -1.0, volume).max(axis=0)
        m_margin = (best_s - second) < min_margin

    # 2) controllo left-right: la disparity vista dalla destra deve concordare
    if lr_thresh is not None:
        volR = np.full_like(volume, -1.0)
        for i, d in enumerate(range(min_disp, max_disp + 1)):
            if d >= W:
                break
            volR[i, :, : W - d] = volume[i, :, d:]
        dR = np.argmax(volR, axis=0) + min_disp
        xr = np.clip(np.arange(W)[None, :] - disp_int, 0, W - 1)
        dR_at = np.take_along_axis(dR, xr, axis=1)
        m_lr = np.abs(dR_at - disp_int) > lr_thresh

    reject = m_tex | m_nocand | m_score | m_margin | m_lr
    disp[reject] = np.nan

    status = np.full((H, W), ST_OK, np.uint8)
    for code, m in ((ST_LR, m_lr), (ST_MARGIN, m_margin), (ST_SCORE, m_score), (ST_NOCAND, m_nocand), (ST_TEXTURE, m_tex)):
        status[m] = code
    return disp, best_s, status


# --------------------------------------------------------------------------- #
# Metriche e visualizzazione
# --------------------------------------------------------------------------- #


def compute_metrics(d_ncc, d_ref, score, min_score, min_disp, max_disp):
    valid_ref = np.isfinite(d_ref) & (d_ref > min_disp) & (d_ref < max_disp)
    valid_ncc = np.isfinite(d_ncc) & (score >= min_score)
    both = valid_ref & valid_ncc
    n_both = int(both.sum())
    out = {
        "cover_ref": valid_ref.mean(),
        "cover_ncc_on_ref": (valid_ncc & valid_ref).sum() / max(valid_ref.sum(), 1),
        "n_pixels": n_both,
    }
    if n_both == 0:
        out.update(mae=np.nan, rmse=np.nan, median=np.nan, bad1=np.nan, bad2=np.nan, bad3=np.nan)
        return out, both, np.zeros_like(d_ref)
    err = np.abs(d_ncc - d_ref)
    e = err[both]
    out.update(
        mae=float(e.mean()),
        rmse=float(np.sqrt((e ** 2).mean())),
        median=float(np.median(e)),
        bad1=float((e > 1).mean() * 100),
        bad2=float((e > 2).mean() * 100),
        bad3=float((e > 3).mean() * 100),
    )
    return out, both, err


def colorize(d, vmax, mask=None):
    x = np.nan_to_num(d, nan=0.0)
    img = cv2.applyColorMap(np.clip(x / vmax * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    bad = ~np.isfinite(d) if mask is None else ~mask
    img[bad] = 0
    return img

# colori BGR del pannello diagnostico
DIAG_COLORS = {
    "ok_good": (60, 200, 60),       # accettato, errore <= soglia vs ZED
    "ok_bad": (40, 40, 230),        # accettato, errore > soglia vs ZED
    "ok_noref": (230, 160, 40),     # accettato, ma la ZED non ha valore
    ST_TEXTURE: (110, 110, 110),
    ST_NOCAND: (35, 35, 35),
    ST_SCORE: (0, 215, 240),
    ST_MARGIN: (0, 140, 255),
    ST_LR: (220, 0, 220)
}

def diagnostic_image(status, d_ncc, d_ref, good_px):
    """Pannello con un colore per ogni esito del matching. Ritorna (img, conteggi)."""
    H, W = status.shape
    img = np.zeros((H, W, 3), np.uint8)
    for code in STATUS_NAMES:
        img[status == code] = DIAG_COLORS[code]
    ok = status == ST_OK
    has_ref = np.isfinite(d_ref)
    err = np.abs(d_ncc - d_ref)
    good = ok & has_ref & (err <= good_px)
    bad = ok & has_ref & (err > good_px)
    noref = ok & ~has_ref
    img[good] = DIAG_COLORS["ok_good"]
    img[bad] = DIAG_COLORS["ok_bad"]
    img[noref] = DIAG_COLORS["ok_noref"]
    n = H * W
    entries = [
        ("ok_good", f"accepted, err<={good_px:g}px", good.sum()),
        ("ok_bad", f"accepted, err>{good_px:g}px", bad.sum()),
        ("ok_noref", "accepted, no ZED ref", noref.sum()),
    ] + [(c, STATUS_NAMES[c], (status == c).sum()) for c in STATUS_NAMES]
    return img, [(k, label, int(c), 100.0 * c / n) for k, label, c in entries]

def draw_legend(img, entries):
    H, W = img.shape[:2]
    fs = max(0.3, W / 2000.0)
    th = 1 if fs < 0.7 else 2
    lh = int(36 * fs) + 4
    x0, y0 = 8, int(50 * fs) + 12
    box = max(6, int(22 * fs))
    p1 = (x0 - 4, y0 - lh)
    p2 = (x0 + int(520 * fs) + box, y0 + lh * (len(entries) - 1) + 8)
    roi = img[p1[1]:p2[1], p1[0]:p2[0]]
    img[p1[1]:p2[1], p1[0]:p2[0]] = (roi * 0.35).astype(np.uint8)  # sfondo scuro semi-trasparente
    for i, (key, label, _, pct) in enumerate(entries):
        y = y0 + i * lh
        cv2.rectangle(img, (x0, y - box), (x0 + box, y), DIAG_COLORS[key], -1)
        cv2.rectangle(img, (x0, y - box), (x0 + box, y), (255, 255, 255), 1)
        cv2.putText(img, f"{label}  {pct:.1f}%", (x0 + box + 8, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th)

def save_visual(path, left_gray, d_ncc, d_ref, err, mask, vmax, status, good_px=1.0,
                err_max=5.0, save_err=False):
    g = cv2.cvtColor(np.clip(left_gray, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    p_ncc = colorize(d_ncc, vmax)
    p_ref = colorize(d_ref, vmax)
    p_diag, entries = diagnostic_image(status, d_ncc, d_ref, good_px)

    fs = max(0.45, g.shape[1] / 1200.0)
    for img, t in ((g, "left"), (p_ncc, "NCC"), (p_ref, "ZED"), (p_diag, "NCC diagnostic")):
        cv2.putText(img, t, (10, int(32 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2)
    draw_legend(p_diag, entries)

    top = np.hstack([g, p_ncc])
    bot = np.hstack([p_ref, p_diag])
    cv2.imwrite(str(path), np.vstack([top, bot]))

    if save_err:
        e = np.where(mask, err, 0)
        p_err = cv2.applyColorMap(np.clip(e / err_max * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        p_err[~mask] = 0
        cv2.imwrite(str(Path(path).with_name(Path(path).stem + "_err.png")), p_err)
    return entries
# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def norm(t):
    return t.lstrip("/")


def main():
    ap = argparse.ArgumentParser(description="NCC vs disparity ZED da rosbag2 (senza ROS)")
    ap.add_argument("bag", help="cartella della rosbag2 (o file .mcap/.db3)")
    ap.add_argument("--left", default="zed/zed_node/left/image_rect_color")
    ap.add_argument("--right", default="zed/zed_node/right/image_rect_color")
    ap.add_argument("--disp", default="zed/zed_node/disparity/disparity_image")
    ap.add_argument("--window", type=int, default=15, help="finestra NCC (dispari)")
    ap.add_argument("--scale", type=float, default=0.25, help="fattore di riduzione immagini")
    ap.add_argument("--max-disp", type=float, default=128,
                    help="disparity massima a risoluzione piena (default: da messaggio ZED, max 256)")
    ap.add_argument("--min-disp", type=float, default=0.0)
    ap.add_argument("--min-score", type=float, default=0.85, help="soglia NCC per considerare valido il pixel")
    ap.add_argument("--min-texture", type=float, default=5.0,
                    help="varianza minima locale (livelli di grigio^2) per fidarsi del match; prova 25-100")
    ap.add_argument("--min-margin", type=float, default=0.003,
                    help="differenza minima tra miglior e secondo miglior score NCC (es. 0.05)")
    ap.add_argument("--lr-thresh", type=float, default=None,
                    help="abilita il controllo left-right con tolleranza in px (es. 1.0)")
    ap.add_argument("--no-subpixel", action="store_true")
    ap.add_argument("--every", type=int, default=1, help="elabora 1 frame ogni N")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = tutti")
    ap.add_argument("--save-vis-every", type=int, default=10, help="salva immagine ogni N frame elaborati")
    ap.add_argument("--disp-sign", choices=["auto", "pos", "neg"], default="auto",
                    help="segno della disparity ZED: auto (default), pos, neg (inverte)")
    ap.add_argument("--distro", default="ROS2_HUMBLE",
                    help="typestore di default per bag senza definizioni (es. ROS2_FOXY, ROS2_HUMBLE, ROS2_JAZZY)")
    ap.add_argument("--out", default="ncc_results")
    ap.add_argument("--diag-good-px", type=float, default=1.0,
                    help="soglia (px) sotto cui un pixel accettato e' 'corretto' nel pannello diagnostico")
    ap.add_argument("--save-err", action="store_false",
                    help="salva anche la mappa dell'errore |NCC-ZED| come frame_XXXX_err.png")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    want = {norm(args.left): "left", norm(args.right): "right", norm(args.disp): "disp"}
    pending = {}  # stamp_ns -> {"left":..., "right":..., "disp":...}
    rows = []
    seen = 0
    processed = 0

    typestore = get_typestore(getattr(Stores, args.distro))
    with AnyReader([Path(args.bag)], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if norm(c.topic) in want]
        found = {norm(c.topic) for c in conns}
        missing = set(want) - found
        if missing:
            avail = "\n  ".join(c.topic for c in reader.connections)
            raise SystemExit(f"Topic non trovati: {missing}\nDisponibili:\n  {avail}")

        for conn, ts, raw in reader.messages(connections=conns):
            kind = want[norm(conn.topic)]
            msg = reader.deserialize(raw, conn.msgtype)
            st = stamp_ns(msg)
            slot = pending.setdefault(st, {})
            slot[kind] = msg
            if len(slot) < 3:
                # pulizia timestamp vecchi mai completati
                if len(pending) > 50:
                    for k in sorted(pending)[:-20]:
                        pending.pop(k)
                continue

            pending.pop(st)
            seen += 1
            if (seen - 1) % args.every:
                continue

            # ---- decodifica ----
            l_arr, l_enc = image_msg_to_numpy(slot["left"])
            r_arr, r_enc = image_msg_to_numpy(slot["right"])
            L = to_gray_float(l_arr, l_enc)
            R = to_gray_float(r_arr, r_enc)
            dm = slot["disp"]
            d_ref, _ = image_msg_to_numpy(dm.image)
            d_ref = d_ref.astype(np.float32)
            d_ref[~np.isfinite(d_ref)] = np.nan
            fin = d_ref[np.isfinite(d_ref)]
            if args.disp_sign == "neg" or (args.disp_sign == "auto" and fin.size and np.median(fin) < 0):
                d_ref = -d_ref  # alcune versioni del wrapper ZED pubblicano disparity negativa

            s = args.scale
            if s != 1.0:
                L = cv2.resize(L, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                R = cv2.resize(R, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                d_ref = cv2.resize(d_ref, (L.shape[1], L.shape[0]), interpolation=cv2.INTER_NEAREST) * s

            msg_max = float(dm.max_disparity)
            if not np.isfinite(msg_max) or msg_max < 16:
                msg_max = 128.0  # valore nel messaggio non affidabile
            full_max = args.max_disp if args.max_disp else min(msg_max, 256.0)
            max_d = int(np.ceil(full_max * s))
            min_d = int(np.floor(args.min_disp * s))

            # ---- NCC ----
            d_ncc, score, status = ncc_disparity(
                L, R, max_d, win=args.window, min_disp=min_d, min_score=args.min_score, subpixel=not args.no_subpixel,
                min_texture=args.min_texture, min_margin=args.min_margin, lr_thresh=args.lr_thresh
            )

            if processed == 0:
                fr = d_ref[np.isfinite(d_ref)]
                print("[diag] ZED disp: enc=%s shape=%s finite=%.1f%%" % (dm.image.encoding, d_ref.shape, 100 * np.isfinite(d_ref).mean()))
                if fr.size:
                    print("[diag] ZED disp: min=%.2f median=%.2f max=%.2f | msg max/min_disparity=%.2f/%.2f"
                          % (fr.min(), np.median(fr), fr.max(), dm.min_disparity*s, dm.max_disparity*s))
                else:
                    print("[diag] ZED disp: NESSUN valore finito -> la disparity ZED nella bag e' vuota")
                print("[diag] NCC: finite=%.1f%% score p50=%.2f p90=%.2f | range usato %d..%d px"
                      % (100 * np.isfinite(d_ncc).mean(), np.nanpercentile(score, 50), np.nanpercentile(score, 90), min_d, max_d))
                print("[diag] left gray: min=%.0f max=%.0f std=%.1f" % (L.min(), L.max(), L.std()))

            m, mask, err = compute_metrics(d_ncc, d_ref, score, args.min_score, min_d, max_d)
            m.update(frame=processed, stamp_ns=st)
            rows.append(m)
            processed += 1
            print(
                f"[{processed:4d}] t={st*1e-9:.3f} MAE={m['mae']:.3f} RMSE={m['rmse']:.3f} "
                f"bad1={m['bad1']:.1f}% bad3={m['bad3']:.1f}% px={m['n_pixels']}"
            )

            for code, name in STATUS_NAMES.items():
                m["frac_" + name.replace(" ", "_").replace("-", "")] = float((status == code).mean())
            m["frac_accepted"] = float((status == ST_OK).mean())

            if args.save_vis_every and (processed - 1) % args.save_vis_every == 0:
                save_visual(out_dir / f"frame_{processed-1:04d}.png", L, d_ncc, d_ref, err, mask, max_d, status)

            if args.max_frames and processed >= args.max_frames:
                break

    if not rows:
        raise SystemExit("Nessun frame completo (left+right+disparity con stesso timestamp).")

    keys = ["frame", "stamp_ns", "mae", "rmse", "median", "bad1", "bad2", "bad3",
            "cover_ref", "cover_ncc_on_ref", "n_pixels"]
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows({k: r[k] for k in keys} for r in rows)

    print("\n=== Riepilogo (media sui frame) ===")
    for k in ["mae", "rmse", "median", "bad1", "bad2", "bad3", "cover_ncc_on_ref"]:
        print(f"{k:18s} {np.nanmean([r[k] for r in rows]):.3f}")
    print(f"\nRisultati in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
