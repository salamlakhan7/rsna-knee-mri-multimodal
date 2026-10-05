"""DICOM zip -> 3 planes x 16 slices x 192 px (same preprocessing as training) -> ResNet18 + attention pooling."""
import os, re, zipfile
import numpy as np, pydicom, cv2

S, K = 192, 16
PLANES = ["Sagittal", "Coronal", "Axial"]
MAX_FILES, MAX_BYTES = 6000, 1_500_000_000
FLUID = re.compile(r"t2|\bpd|pd[-_ ]|proton|stir|tirm|dess|water|\bfs\b|fat ?sat|spair|spir|dixon", re.I)
T1 = re.compile(r"t1", re.I)

def extract_zip(file_obj, dest):
    with zipfile.ZipFile(file_obj) as z:
        infos = z.infolist()
        if len(infos) > MAX_FILES: raise ValueError(f"Zip has more than {MAX_FILES} files.")
        if sum(i.file_size for i in infos) > MAX_BYTES: raise ValueError("Zip is too large when extracted.")
        base = os.path.realpath(dest)
        for i in infos:
            t = os.path.realpath(os.path.join(dest, i.filename))
            if t != base and not t.startswith(base + os.sep): raise ValueError("Unsafe path inside zip.")
        z.extractall(dest)

def plane_from_iop(iop):
    n = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
    return PLANES[int(np.argmax(np.abs(n)))]          # x normal = sagittal, y = coronal, z = axial

def scan_series(root):
    series = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(dp, f)
            try:
                h = pydicom.dcmread(p, stop_before_pixels=True)
                uid = str(h.SeriesInstanceUID); int(h.Rows)
            except Exception:
                continue
            e = series.setdefault(uid, {"files": [], "plane": None, "desc": "", "fluid": False})
            e["files"].append(p)
            if e["plane"] is None:
                try: e["plane"] = plane_from_iop([float(x) for x in h.ImageOrientationPatient])
                except Exception: pass
                d = f"{getattr(h, 'SeriesDescription', '')} {getattr(h, 'ProtocolName', '')}".strip()
                e["desc"] = d
                e["fluid"] = bool(FLUID.search(d)) and not (T1.search(d) and not re.search(r"t2|pd", d, re.I))
    return {u: e for u, e in series.items() if e["plane"] and len(e["files"]) >= 8}

def pick_series(series):
    picks = []
    for plane in PLANES:
        c = [(u, e) for u, e in series.items() if e["plane"] == plane]
        use = [x for x in c if x[1]["fluid"]] or c
        use.sort(key=lambda x: -len(x[1]["files"]))
        picks.append(use[0][0] if use else None)
    return picks

def _key(h):
    try:
        ipp = np.array(h.ImagePositionPatient, float); iop = np.array(h.ImageOrientationPatient, float)
        return float(np.dot(ipp, np.cross(iop[:3], iop[3:])))
    except Exception:
        pass
    try: return float(h.InstanceNumber)
    except Exception: return 0.0

def load_series(paths):
    keys = sorted((_key(pydicom.dcmread(p, stop_before_pixels=True)), p) for p in paths)
    n = len(keys); lo = int(0.2 * n); hi = max(int(0.8 * n), lo + 1)
    idx = np.clip(np.linspace(lo, hi - 1, K).round().astype(int), 0, n - 1)
    sl = []
    for i in idx:
        ds = pydicom.dcmread(keys[i][1]); a = ds.pixel_array.astype(np.float32)
        if a.ndim == 3: a = a[..., 0] if a.shape[-1] in (3, 4) else a[0]
        a = a * float(getattr(ds, "RescaleSlope", 1) or 1) + float(getattr(ds, "RescaleIntercept", 0) or 0)
        sl.append(cv2.resize(a, (S, S), interpolation=cv2.INTER_AREA if a.shape[0] > S else cv2.INTER_LINEAR))
    v = np.stack(sl); p1, p99 = np.percentile(v, [1, 99])
    return (np.clip((v - p1) / max(p99 - p1, 1e-6), 0, 1) * 255).astype(np.uint8)

def build_study(series, picks):
    arr = np.zeros((3, K, S, S), np.uint8); used = []
    for pi, uid in enumerate(picks):
        if uid is None: continue
        arr[pi] = load_series(series[uid]["files"])
        used.append({"plane": PLANES[pi], "description": series[uid]["desc"] or "(none)",
                     "slices": len(series[uid]["files"]), "fluid_sensitive": series[uid]["fluid"]})
    return arr, used

def load_model(weights_path):
    import torch, torch.nn as nn, timm
    class KneeNet(nn.Module):
        def __init__(self, backbone, d):
            super().__init__()
            self.bb, self.att = backbone, nn.Linear(d, 1)
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(3 * d, 12))
        def forward(self, x):
            B, P, Ks, H, W = x.shape
            f = self.bb(x.reshape(B * P * Ks, 1, H, W)).view(B, P, Ks, -1)
            w = torch.softmax(self.att(f).squeeze(-1), dim=2)
            return self.head((w.unsqueeze(-1) * f).sum(2).reshape(B, -1))
    m = KneeNet(timm.create_model("resnet18", pretrained=False, in_chans=1, num_classes=0), 512)
    m.load_state_dict(torch.load(weights_path, map_location="cpu")); m.eval(); return m

def predict(model, arr):
    import torch
    x = (torch.from_numpy(arr)[None].float() / 255 - 0.45) / 0.225
    with torch.no_grad():
        return torch.sigmoid(model(x).float()).numpy()[0]
