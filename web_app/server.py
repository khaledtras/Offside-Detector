import base64
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

APP_DIR = Path(__file__).resolve().parent


def _resolve_model_path() -> Path:
    candidates = []
    if os.environ.get("OFFSIDE_MODEL_PATH"):
        candidates.append(Path(os.environ["OFFSIDE_MODEL_PATH"]))
    candidates.extend([
        APP_DIR.parent / "offside_pytorch_model.pt",
        APP_DIR.parent / "notebooks" / "offside_pytorch_model.pt",
        Path("/content/drive/MyDrive/Offside_Detector/v2_var/offside_pytorch_model.pt"),
    ])
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "Model 'offside_pytorch_model.pt' not found in: " + ", ".join(map(str, candidates))
    )


MODEL_PATH = _resolve_model_path()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
det_model = YOLO("yolov8x.pt")
pose_model = YOLO("yolov8n-pose.pt")


class ResidualOffsideNet(nn.Module):
    def __init__(self, ctx_dim: int = 6):
        super().__init__()
        self.physics_head = nn.Linear(1, 1)
        self.mlp = nn.Sequential(
            nn.Linear(ctx_dim + 1, 16),
            nn.SiLU(),
            nn.Dropout(0.35),
            nn.Linear(16, 8),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(8, 1),
        )

    def forward(self, phys_x: torch.Tensor, ctx_x: torch.Tensor) -> torch.Tensor:
        phys_logit = self.physics_head(phys_x)
        full_in = torch.cat([phys_x, ctx_x], dim=1)
        corr_logit = 0.5 * torch.tanh(self.mlp(full_in))  # Cap MLP delta to prevent overfit
        return (phys_logit + corr_logit).squeeze(-1)


ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
pt_model = ResidualOffsideNet(ctx_dim=6).to(device)
pt_model.load_state_dict(ckpt["model_state_dict"])
pt_model.eval()

SCALER_MEAN = ckpt["scaler_mean"]
SCALER_SCALE = ckpt["scaler_scale"]

LEGAL_KP = [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]  # Exclude arms per FIFA Law 11
GRASS_LOWER = np.array([36, 40, 40], dtype=np.uint8)       # Preserve yellow kits
GRASS_UPPER = np.array([85, 255, 255], dtype=np.uint8)
COLOR_WEIGHTS = np.array([0.50, 1.40, 1.40], dtype=np.float32)  # De-emphasize luminance L


def get_pitch_mask(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    raw = cv2.inRange(hsv, np.array([30, 40, 40], dtype=np.uint8), GRASS_UPPER)
    closed = cv2.morphologyEx(
        raw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((h, w), dtype=np.uint8)
    if contours:
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        cv2.drawContours(mask, [hull], -1, 255, thickness=cv2.FILLED)
    else:
        mask[int(h * 0.2) :, :] = 255
    return cv2.erode(mask, np.ones((45, 15), dtype=np.uint8), iterations=1)  # Trim ad boards


def detect_players_and_ball(img: np.ndarray, pitch_mask: np.ndarray, imgsz: int = 1280):
    h, w = img.shape[:2]
    det = det_model(img, classes=[0, 32], conf=0.10, imgsz=imgsz, verbose=False)[0]
    person_boxes, ball_candidates = [], []

    if det.boxes is not None and len(det.boxes) > 0:
        for b, cls_id, conf in zip(
            det.boxes.xyxy.cpu().numpy(),
            det.boxes.cls.cpu().numpy().astype(int),
            det.boxes.conf.cpu().numpy(),
        ):
            if cls_id == 0:
                person_boxes.append(b)
            elif cls_id == 32 and conf >= 0.28:
                bx, by = int((b[0] + b[2]) / 2.0), int((b[1] + b[3]) / 2.0)
                bw, bh = b[2] - b[0], b[3] - b[1]
                ar = bw / max(bh, 1e-5)
                if (
                    0 <= bx < w
                    and 0 <= by < h
                    and pitch_mask[by, bx] > 0
                    and 0.6 <= ar <= 1.6
                    and max(bw, bh) <= 45
                ):
                    ball_candidates.append(
                        (conf, np.array([float(bx), float(by)], dtype=np.float32))
                    )

    ball_pos = (
        sorted(ball_candidates, key=lambda x: x[0], reverse=True)[0][1]
        if ball_candidates
        else None
    )

    players = []
    for box in person_boxes:
        cx = int(np.clip((box[0] + box[2]) / 2.0, 0, w - 1))
        fy = int(np.clip(box[3] - 2, 0, h - 1))
        if pitch_mask[fy, cx] == 0:  # Foot-only check rejects sidelines
            continue
        x1, y1, x2, y2 = box.astype(int)
        bw, bh = x2 - x1, y2 - y1
        cx1, cy1 = max(0, x1 - int(0.10 * bw)), max(0, y1 - int(0.10 * bh))
        cx2, cy2 = min(w, x2 + int(0.10 * bw)), min(h, y2 + int(0.10 * bh))
        crop = img[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        pose_conf = 0.15
        if crop.shape[0] < 120:  # Upscale far players for Pose
            scale = 3 if crop.shape[0] < 60 else 2
            crop = cv2.resize(
                crop,
                (crop.shape[1] * scale, crop.shape[0] * scale),
                interpolation=cv2.INTER_LINEAR,
            )
            pose_conf = 0.10

        pres = pose_model(crop, conf=pose_conf, verbose=False)[0]
        got_kp = (
            pres.keypoints is not None
            and len(pres.keypoints) > 0
            and pres.keypoints.conf is not None
        )
        if got_kp:
            confs_all = pres.keypoints.conf.cpu().numpy()
            best_k = int(np.argmax(confs_all.mean(axis=1)))
            sx, sy = (cx2 - cx1) / crop.shape[1], (cy2 - cy1) / crop.shape[0]
            g_xy = (
                pres.keypoints.xy.cpu().numpy()[best_k] * np.array([sx, sy])
                + np.array([cx1, cy1])
            )
            k_conf = confs_all[best_k]
            pts = [g_xy[idx] for idx in LEGAL_KP if k_conf[idx] > 0.25]
            if pts:
                best_pt = max(pts, key=lambda pt: pt[0])
                max_x, max_pt = float(best_pt[0]), best_pt
            else:
                max_x, max_pt = float(x2), np.array([float(x2), (y1 + y2) / 2.0])
            players.append(
                {
                    "box": box,
                    "keypoints_xy": g_xy,
                    "keypoints_conf": k_conf,
                    "max_x": max_x,
                    "max_point": max_pt,
                    "box_height": float(bh),
                    "center_x": (x1 + x2) / 2.0,
                    "center_y": (y1 + y2) / 2.0,
                }
            )
        else:
            max_x = float(box[2]) - 0.10 * float(bw)
            players.append(
                {
                    "box": box,
                    "keypoints_xy": None,
                    "keypoints_conf": None,
                    "max_x": max_x,
                    "max_point": np.array([max_x, (y1 + y2) / 2.0]),
                    "box_height": float(bh),
                    "center_x": (x1 + x2) / 2.0,
                    "center_y": (y1 + y2) / 2.0,
                }
            )
    return players, ball_pos


def rgb_to_wlab(rgb) -> np.ndarray:
    bgr = np.uint8([[[int(rgb[2]), int(rgb[1]), int(rgb[0])]]])
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)[0, 0].astype(np.float32) * COLOR_WEIGHTS


def get_jersey_lab(img: np.ndarray, p: dict) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = p["box"].astype(int)
    bh, bw = y2 - y1, x2 - x1
    done = False
    if p["keypoints_xy"] is not None and p["keypoints_conf"] is not None:
        pts = [
            p["keypoints_xy"][i] for i in [5, 6, 11, 12] if p["keypoints_conf"][i] > 0.25
        ]
        if len(pts) >= 2:
            cx = float(np.mean([q[0] for q in pts]))
            cy = float(np.mean([q[1] for q in pts]))
            hs = int(np.clip(0.11 * bh, 6, 30))
            rx1, rx2 = max(0, int(cx - hs)), min(w, int(cx + hs))
            ry1, ry2 = max(0, int(cy - hs)), min(h, int(cy + hs))
            if rx2 > rx1 and ry2 > ry1:
                done = True
    if not done:
        ry1, ry2 = max(0, y1 + int(0.20 * bh)), min(h, y1 + int(0.46 * bh))
        rx1, rx2 = max(0, x1 + int(0.28 * bw)), min(w, x1 + int(0.72 * bw))
    crop = img[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        return np.array([50.0, 0.0, 0.0], dtype=np.float32)
    gm = cv2.inRange(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV), GRASS_LOWER, GRASS_UPPER)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)
    ng = cv2.bitwise_not(gm)
    if gm.sum() / 255.0 / max(crop.shape[0] * crop.shape[1], 1) < 0.70 and (ng > 0).sum() >= 4:
        return np.median(lab[ng > 0], axis=0).astype(np.float32)
    return np.median(lab.reshape(-1, 3), axis=0).astype(np.float32)


def classify_and_predict_var(img: np.ndarray, cal: dict):
    h, w = img.shape[:2]
    mask = get_pitch_mask(img)
    players, ball_pos = detect_players_and_ball(img, mask)
    if len(players) < 2:
        raise ValueError(f"Too few pitch players detected ({len(players)}).")

    att_lab, def_lab = rgb_to_wlab(cal["att_rgb"]), rgb_to_wlab(cal["def_rgb"])
    attackers, defenders, excluded = [], [], []
    for p in players:
        c = get_jersey_lab(img, p) * COLOR_WEIGHTS
        da = float(np.linalg.norm(c - att_lab))
        dd = float(np.linalg.norm(c - def_lab))
        p["d_att"], p["d_def"], p["team_dist"] = da, dd, min(da, dd)
        if min(da, dd) < 44.0 and min(da, dd) / (max(da, dd) + 1e-6) < 0.80:
            (attackers if da <= dd else defenders).append(p)
        else:
            excluded.append(p)

    if not attackers and excluded:
        pick = min(excluded, key=lambda p: p["d_att"])
        attackers.append(pick)
        excluded = [x for x in excluded if x is not pick]
    if not defenders and excluded:
        pick = min(excluded, key=lambda p: p["d_def"])
        defenders.append(pick)
        excluded = [x for x in excluded if x is not pick]

    if not attackers or not defenders:
        raise ValueError("Could not separate both attacking and defending teams.")

    gk_cands = [
        (p["max_x"], i)
        for i, p in enumerate(excluded)
        if p["max_x"] / w >= 0.65
        and 0.08 <= p["center_y"] / h <= 0.92
        and p["d_att"] >= 50.0  # Require distinct kit from both teams
        and p["d_def"] >= 50.0
    ]
    goalkeeper = excluded.pop(max(gk_cands)[1]) if gk_cands else None

    if goalkeeper is None and len(attackers) >= 2 and defenders:
        mx_d = max(p["max_x"] for p in defenders)
        ra = max(attackers, key=lambda p: p["max_x"])
        if ra["max_x"] > mx_d + 0.08 * w and ra["max_x"] >= 0.62 * w and ra["d_att"] > 26.0:
            attackers = [p for p in attackers if p is not ra]  # Identity guard avoids NumPy Eq
            goalkeeper = ra

    s = float(cal["pitch_slope"])
    proj = lambda p, ry: float(p["max_x"] + s * (ry - float(p["box"][3])))

    opp = list(defenders) + ([goalkeeper] if goalkeeper is not None else [])
    ref_y = float(np.mean([p["box"][3] for p in opp]))
    opp_s = sorted(opp, key=lambda p: proj(p, ref_y), reverse=True)
    k = 1 if (goalkeeper is not None and len(opp_s) >= 2) else 0
    d1 = opp_s[k]
    d2 = opp_s[k + 1] if len(opp_s) > k + 1 else d1
    d1_fy = float(d1["box"][3])

    # FIFA Law 11: max(2nd-last defender, ball)
    ball_is_line = False
    ref_line_x, ref_line_y, ref_line_lbl = float(d1["max_x"]), d1_fy, "DEF LINE"
    cand_attackers = list(attackers)

    if ball_pos is not None:
        ball_proj_x = float(ball_pos[0] + s * (d1_fy - float(ball_pos[1])))
        if ball_proj_x > float(d1["max_x"]):
            ball_is_line = True
            ref_line_x, ref_line_y = float(ball_pos[0]), float(ball_pos[1])
            ref_line_lbl = "BALL OFFSIDE LINE"
            if len(cand_attackers) >= 2:  # Exclude passer in 2v1 breakaway
                passer = min(
                    cand_attackers,
                    key=lambda p: np.hypot(p["center_x"] - ball_pos[0], p["box"][3] - ball_pos[1]),
                )
                if np.hypot(passer["center_x"] - ball_pos[0], passer["box"][3] - ball_pos[1]) < 0.08 * w:
                    cand_attackers = [p for p in cand_attackers if p is not passer]

    att_s = sorted(cand_attackers, key=lambda p: proj(p, ref_line_y), reverse=True)
    a1 = att_s[0]
    a2 = att_s[1] if len(att_s) > 1 else a1

    var_diff_x = (proj(a1, ref_line_y) - ref_line_x) / w
    sec_diff_x = (proj(a2, ref_line_y) - ref_line_x) / w
    spread_x = (float(d1["max_x"]) - proj(d2, d1_fy)) / w
    y_diff = (float(a1["box"][3]) - ref_line_y) / h
    scale_r = a1["box_height"] / (d1["box_height"] + 1e-6)
    att_x_n = float(a1["max_x"]) / w
    ball_diff = ((float(a1["max_x"]) - ball_pos[0]) / w) if ball_pos is not None else 0.0

    phys_v = np.array([[np.tanh(var_diff_x / 0.05)]], dtype=np.float32)  # Preserve geometric 0.0
    ctx_v = np.array(
        [[np.tanh(sec_diff_x / 0.12), y_diff, np.clip(scale_r, 0.5, 2.0), spread_x, att_x_n, np.tanh(ball_diff / 0.15)]],
        dtype=np.float32,
    )
    ctx_v = (ctx_v - SCALER_MEAN) / (SCALER_SCALE + 1e-8)

    with torch.no_grad():
        logit = pt_model(torch.tensor(phys_v, device=device), torch.tensor(ctx_v, device=device))
        prob_offside = float(torch.sigmoid(logit).item())

    decision = "OFFSIDE" if prob_offside >= 0.5 else "ONSIDE"

    vis = img.copy()
    for p in attackers:
        b = p["box"].astype(int)
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cv2.circle(vis, tuple(p["max_point"].astype(int)), 5, (0, 0, 255), -1)
    for p in defenders:
        b = p["box"].astype(int)
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)
        cv2.circle(vis, tuple(p["max_point"].astype(int)), 5, (255, 0, 0), -1)
    if goalkeeper is not None:
        b = goalkeeper["box"].astype(int)
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 3)
        cv2.putText(vis, "GK", (b[0], max(20, b[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    ref_col = (255, 0, 200) if ball_is_line else (255, 0, 0)
    for x0, y0, col, lbl, ly in [
        (float(a1["max_x"]), float(a1["box"][3]), (0, 0, 255), "ATT LINE", 35),
        (ref_line_x, ref_line_y, ref_col, ref_line_lbl, 65),
    ]:
        p1 = (int(round(x0 + s * (-2 * h - y0))), -2 * h)
        p2 = (int(round(x0 + s * (3 * h - y0))), 3 * h)
        ok, c1, c2 = cv2.clipLine((0, 0, w, h), p1, p2)  # Exact clip avoids edge bending
        if ok:
            cv2.line(vis, c1, c2, col, 3)
            lx = int(np.clip(c1[0] + 6, 8, max(8, w - 165)))
            ly_draw = int(np.clip(c1[1] + ly, 25, max(25, h - 20)))
            cv2.putText(vis, lbl, (lx, ly_draw), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

    if ball_pos is not None:
        cv2.circle(vis, (int(ball_pos[0]), int(ball_pos[1])), 8, (255, 0, 255), -1)

    return vis, {
        "decision": decision,
        "prob_offside": prob_offside,
        "var_diff_x": float(var_diff_x),
        "n_att": len(attackers),
        "n_def": len(defenders),
        "gk": goalkeeper is not None,
        "ball_is_line": ball_is_line,
    }


def _sample_rgb_norm(img_bgr: np.ndarray, nx: float, ny: float, half: int = 3):
    h, w = img_bgr.shape[:2]
    px = int(np.clip(round(nx * w), 0, w - 1))
    py = int(np.clip(round(ny * h), 0, h - 1))
    patch = img_bgr[max(0, py - half) : min(h, py + half + 1), max(0, px - half) : min(w, px + half + 1)]
    med = np.median(patch.reshape(-1, 3), axis=0)
    return (int(med[2]), int(med[1]), int(med[0]))


def _encode_b64_jpeg(img_bgr: np.ndarray, quality: int = 92) -> str:
    _, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


app = FastAPI(title="OffsideX VAR API")
app.mount("/assets", StaticFiles(directory=str(APP_DIR / "assets")), name="assets")


@app.get("/")
def serve_home():
    return FileResponse(APP_DIR / "index.html")


@app.get("/api/health")
def api_health():
    return {
        "ok": True,
        "device": str(device),
        "oof_accuracy": float(ckpt.get("oof_accuracy", 0.0)),
        "oof_f1": float(ckpt.get("oof_f1", 0.0)),
    }


@app.post("/api/analyze")
async def api_analyze(
    image: UploadFile = File(...),
    att_nx: float = Form(...),
    att_ny: float = Form(...),
    def_nx: float = Form(...),
    def_ny: float = Form(...),
    l1_nx: Optional[float] = Form(None),
    l1_ny: Optional[float] = Form(None),
    l2_nx: Optional[float] = Form(None),
    l2_ny: Optional[float] = Form(None),
    flipped: bool = Form(False),
    zero_slope: bool = Form(False),
):
    raw_bytes = await image.read()
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    raw_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if raw_bgr is None:
        return JSONResponse(status_code=400, content={"ok": False, "error": "Could not decode image file."})

    aligned_bgr = cv2.flip(raw_bgr, 1) if flipped else raw_bgr.copy()
    h, w = aligned_bgr.shape[:2]

    att_rgb = _sample_rgb_norm(aligned_bgr, att_nx, att_ny)
    def_rgb = _sample_rgb_norm(aligned_bgr, def_nx, def_ny)

    if zero_slope or l1_nx is None or l1_ny is None or l2_nx is None or l2_ny is None:
        slope = 0.0
    else:
        dx = (l2_nx - l1_nx) * w
        dy = (l2_ny - l1_ny) * h
        slope = float(np.clip(dx / dy, -20.0, 20.0)) if abs(dy) > 5.0 else 0.0

    cal = {"att_rgb": att_rgb, "def_rgb": def_rgb, "pitch_slope": slope}

    try:
        vis_bgr, res = classify_and_predict_var(aligned_bgr, cal)
    except Exception as exc:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(exc)})

    prob_off = float(res["prob_offside"])
    conf_pct = (prob_off * 100.0) if res["decision"] == "OFFSIDE" else ((1.0 - prob_off) * 100.0)

    return {
        "ok": True,
        "decision": res["decision"],
        "prob_offside": round(prob_off, 4),
        "confidence": round(conf_pct, 1),
        "var_diff_x": round(res["var_diff_x"], 4),
        "pitch_slope": round(slope, 3),
        "n_att": res["n_att"],
        "n_def": res["n_def"],
        "gk": bool(res["gk"]),
        "ball_is_line": bool(res["ball_is_line"]),
        "result_b64": _encode_b64_jpeg(vis_bgr),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7865)
