# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 2 — Recoupling Ratio Scan                                        ║
# ║  Teacher : pct_mdca.pth  (FL+MDCA calibrated teacher)                  ║
# ║  Loads   : pct_compare_stats.pt                                         ║
# ║  Saves   : rr_results.pt  |  cell2_rr_heatmap.png                      ║
# ║  NOTE    : pytorch3d replaced with pure-PyTorch equivalents             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import sys, subprocess, os, glob, time, warnings
warnings.filterwarnings("ignore")

def pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

pip("numpy", "matplotlib", "h5py", "tqdm", "POT", "scipy")

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from tqdm import tqdm
import ot as pot

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
torch.manual_seed(42)
np.random.seed(42)

# ── Pure-PyTorch replacements for pytorch3d ops ───────────────────────────────

def sample_farthest_points(xyz, K):
    """
    Farthest Point Sampling — pure PyTorch, no pytorch3d.
    xyz : (B, N, 3)
    Returns (sampled_xyz, indices)
      sampled_xyz : (B, K, 3)
      indices     : (B, K)  int64
    """
    B, N, _ = xyz.shape
    device = xyz.device
    K = min(K, N)

    indices = torch.zeros(B, K, dtype=torch.long, device=device)
    dists   = torch.full((B, N), float("inf"), device=device)
    cur_idx = torch.zeros(B, dtype=torch.long, device=device)

    for i in range(K):
        indices[:, i] = cur_idx
        cur_pt    = xyz[torch.arange(B, device=device), cur_idx].unsqueeze(1)
        new_dists = ((xyz - cur_pt) ** 2).sum(-1)
        dists     = torch.minimum(dists, new_dists)
        cur_idx   = dists.argmax(dim=1)

    sampled_xyz = xyz[torch.arange(B, device=device).unsqueeze(1), indices]
    return sampled_xyz, indices


def knn_points(query, ref, K, return_nn=False):
    """
    K-Nearest Neighbours — pure PyTorch, no pytorch3d.
    query : (B, M, 3)
    ref   : (B, N, 3)
    Returns object with .idx of shape (B, M, K).
    """
    B, M, _ = query.shape
    _, N, _ = ref.shape
    K = min(K, N)

    q2   = (query ** 2).sum(-1, keepdim=True)
    r2   = (ref   ** 2).sum(-1, keepdim=True)
    dot  = torch.bmm(query, ref.transpose(1, 2))
    dist2 = (q2 + r2.transpose(1, 2) - 2 * dot).clamp(min=0)

    _, idx = dist2.topk(K, dim=-1, largest=False, sorted=True)

    class _KNNResult:
        pass
    result = _KNNResult()
    result.idx = idx
    return result

# ── Load from comparison stats (FL+MDCA results) ─────────────────────────────
assert os.path.exists("pct_mdca.pth"),          "Run comparison cell first — pct_mdca.pth not found."
assert os.path.exists("pct_compare_stats.pt"),  "Run comparison cell first — pct_compare_stats.pt not found."

compare_stats = torch.load("pct_compare_stats.pt", weights_only=False)
CFG           = compare_stats["cfg"]
mdca_res      = compare_stats["results"]["FL+MDCA"]

# ── Memory-safe overrides (prevents Linux OOM kill) ───────────────────────────
CFG["num_workers"] = 2      # was 4 — fewer worker processes = less RAM pressure
CFG["batch_size"]  = 32     # was 64 — smaller batches keep the KNN distance
                             #          matrix (B×M×N) from blowing up RAM

print(f"Loaded FL+MDCA teacher | test_acc={mdca_res['test_acc']:.2f}% | "
      f"test_ECE={mdca_res['test_ece']:.2f}% | test_SCE={mdca_res['test_sce']:.4f}e-3")
print(f"Architecture: n_attn={CFG['n_attn']}, embed_dim={CFG['embed_dim']}")
print(f"Memory-safe params: batch_size={CFG['batch_size']}, num_workers={CFG['num_workers']}")

# ── Dataset ────────────────────────────────────────────────────────────────────
DATA_DIR    = "modelnet40_ply_hdf5_2048"
TRAIN_FILES = sorted(glob.glob(os.path.join(DATA_DIR, "ply_data_train*.h5")))
assert TRAIN_FILES, "ModelNet40 not found — re-run comparison cell to download."

def pc_normalize(pc):
    pc = pc - pc.mean(0)
    return pc / (np.sqrt((pc**2).sum(1)).max() + 1e-8)

class MN40(Dataset):
    def __init__(self, files, n_pts=1024):
        self.n = n_pts
        d, l = [], []
        for f in files:
            with h5py.File(f) as h:
                d.append(h["data"][:].astype(np.float32))
                l.append(h["label"][:].flatten().astype(np.int64))
        self.data   = np.concatenate(d)
        self.labels = np.concatenate(l)
        print(f"  Loaded {len(self.data):,} clouds")
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        pts = self.data[i].copy()
        rng = np.random.RandomState(i)
        idx = rng.choice(pts.shape[0], self.n, replace=False)
        return torch.from_numpy(pc_normalize(pts[idx])), int(self.labels[i])

train_pool = MN40(TRAIN_FILES, CFG["n_pts"])
DLK = dict(num_workers=CFG["num_workers"], pin_memory=True, persistent_workers=True)
collect_dl = DataLoader(train_pool, CFG["batch_size"], shuffle=False, drop_last=False, **DLK)

# ── PCT model ─────────────────────────────────────────────────────────────────
class LBR(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(i, o, bias=False), nn.BatchNorm1d(o), nn.ReLU(True))
    def forward(self, x):
        s = x.shape
        return self.net(x.reshape(-1, s[-1])).reshape(*s[:-1], -1)

class SGModule(nn.Module):
    def __init__(self, nout, k, din, dout):
        super().__init__()
        self.nout = nout; self.k = k
        self.l1 = LBR(din * 2, dout); self.l2 = LBR(dout, dout)
    def forward(self, xyz, feat):
        B, N, _ = xyz.shape; no = min(self.nout, N); k = min(self.k, N)
        # ── replaced pytorch3d calls with pure-PyTorch equivalents ──
        sxyz, fi = sample_farthest_points(xyz, K=no)
        ki = knn_points(sxyz, xyz, K=k, return_nn=False).idx
        # ─────────────────────────────────────────────────────────────
        C  = feat.shape[-1]
        nbr = feat.unsqueeze(1).expand(-1, no, -1, -1).gather(
              2, ki.unsqueeze(-1).expand(-1, -1, -1, C))
        ctr = feat.gather(1, fi.unsqueeze(-1).expand(-1, -1, C))
        agg = torch.cat([nbr - ctr.unsqueeze(2).expand_as(nbr),
                         ctr.unsqueeze(2).expand_as(nbr)], -1)
        Bo, No, K, C2 = agg.shape
        out = self.l2(self.l1(agg.reshape(Bo * No * K, C2)))
        return sxyz, out.reshape(Bo, No, K, -1).max(2).values

class NE(nn.Module):
    def __init__(self, ns1, ns2, k, D):
        super().__init__()
        self.lbr = LBR(3, 64); self.sg1 = SGModule(ns1, k, 64, 64); self.sg2 = SGModule(ns2, k, 64, D)
    def forward(self, xyz):
        f = self.lbr(xyz); x2, f2 = self.sg1(xyz, f); x3, f3 = self.sg2(x2, f2); return x3, f3

class OA(nn.Module):
    def __init__(self, D, da):
        super().__init__()
        self.Wq=nn.Linear(D,da,bias=False); self.Wk=nn.Linear(D,da,bias=False)
        self.Wv=nn.Linear(D,D, bias=False); self.lbr=LBR(D,D)
    def forward(self, x):
        Q=self.Wq(x); K=self.Wk(x); V=self.Wv(x)
        scores = torch.bmm(Q, K.transpose(1,2))          # (B, N_query=i, N_key=j)
        A = F.softmax(scores, dim=1)                      # softmax over i (queries), per fixed key j — Eq.9 step 1
        A = A / (A.sum(dim=2, keepdim=True) + 1e-8)        # L1-normalize over j (keys), per fixed query i — Eq.9 step 2
        return self.lbr(x - torch.bmm(A, V)) + x

class PCT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D = cfg["embed_dim"]; da = cfg["da"]; L = cfg["n_attn"]
        self.ne   = NE(cfg["n_sg1"], cfg["n_sg2"], cfg["k_sg"], D)
        self.oas  = nn.ModuleList([OA(D, da) for _ in range(L)])
        self.proj = nn.Linear(D * L, 1024, bias=False)
        self.head = nn.Sequential(
            nn.Linear(2048, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(True), nn.Dropout(cfg["drop_rate"]),
            nn.Linear(256, 256, bias=False),  nn.BatchNorm1d(256), nn.ReLU(True), nn.Dropout(cfg["drop_rate"]),
            nn.Linear(256, cfg["n_classes"]),
        )
    def forward_latents(self, xyz):
        _, h = self.ne(xyz)
        inputs, outputs = [], []
        for oa in self.oas:
            inputs.append(h.detach().clone())
            h = oa(h)
            outputs.append(h.detach().clone())
        return inputs, outputs

# ── Load FL+MDCA calibrated teacher ──────────────────────────────────────────
model = PCT(CFG).to(DEVICE)
model.load_state_dict(torch.load("pct_mdca.pth", map_location=DEVICE))
model.eval()
print(f"\nFL+MDCA calibrated PCT loaded | params = {sum(p.numel() for p in model.parameters()):,}")
print(f"  (This teacher has test_ECE={mdca_res['test_ece']:.2f}% vs CE baseline "
      f"test_ECE={compare_stats['results']['CE']['test_ece']:.2f}%)")

# ── Collect latents ────────────────────────────────────────────────────────────
# Reduced from 80 → 50 batches to cut RAM usage during latent collection.
# With batch_size=32 this still gives 1600 clouds, well above the 128 used for OT.
MAX_BATCHES = 50
L = CFG["n_attn"]
bufs = [[] for _ in range(L)]

print(f"\nCollecting latents from FL+MDCA teacher ({MAX_BATCHES} batches) ...")
t0 = time.perf_counter()
with torch.no_grad():
    for i, (pts, _) in enumerate(tqdm(collect_dl, total=MAX_BATCHES)):
        if i >= MAX_BATCHES: break
        pts = pts.to(DEVICE, non_blocking=True)
        with autocast(DEVICE):
            inputs, _ = model.forward_latents(pts)
        for j, h in enumerate(inputs):
            bufs[j].append(h.cpu().float())

collect_time = time.perf_counter() - t0
latents_in = [torch.cat(b, dim=0) for b in bufs]
N_clouds, N_tok, D_feat = latents_in[0].shape
print(f"  Clouds collected : {N_clouds}  Shape per layer : ({N_clouds}, {N_tok}, {D_feat})")
print(f"  Collect time     : {collect_time:.1f}s")

# ── Recoupling Ratio ──────────────────────────────────────────────────────────
def recoupling_ratio(hm, hn, O_M=128):
    N = hm.shape[0]
    cidx = torch.randperm(N)[:min(O_M, N)]
    a = hm[cidx].float().mean(dim=1).numpy().astype(np.float64)
    b = hn[cidx].float().mean(dim=1).numpy().astype(np.float64)
    O_M_actual = len(a)
    diff = a[:, None, :] - b[None, :, :]
    C = (diff ** 2).sum(-1)
    c_mean = C.mean()
    if c_mean < 1e-12:
        return 0.0
    C = C / c_mean
    wa = np.ones(O_M_actual, dtype=np.float64)
    wb = np.ones(O_M_actual, dtype=np.float64)
    M = pot.emd(wa, wb, C)
    return float(np.clip(1.0 - np.trace(M) / O_M_actual, 0.0, 1.0))

OT_CLOUDS = min(128, N_clouds)
R_mat = np.full((L, L), np.nan)

print(f"\nComputing Recoupling Ratios ({L}×{L} upper triangle, O_M={OT_CLOUDS}) ...")
t_rr = time.perf_counter()
for s in range(L):
    for t in range(s + 1, L):
        R = recoupling_ratio(latents_in[s], latents_in[t], O_M=OT_CLOUDS)
        R_mat[s, t] = R
        tag = ("✓✓ HIGHLY compressible" if R < 0.10 else
               "✓  compressible"         if R < 0.30 else
               "△  moderate"             if R < 0.60 else
               "✗  avoid — heavy crossing")
        print(f"  R[OA{s:2d} → OA{t:2d}] = {R:.6f}   {tag}")
rr_time = time.perf_counter() - t_rr
print(f"\nRR scan complete in {rr_time:.1f}s")

# ── Identify best block ───────────────────────────────────────────────────────
MID_SRC_MIN = 2; MID_SRC_MAX = L - 3; MID_TGT_MAX = L - 2
COMPRESS_SRC = 3; COMPRESS_TGT = 9; R_ZERO_THRESH = 1e-6

mid_pairs = []
for s in range(MID_SRC_MIN, MID_SRC_MAX + 1):
    for t in range(s + 2, MID_TGT_MAX + 1):
        v = R_mat[s, t]
        if not np.isnan(v): mid_pairs.append((s, t, v, t - s))

zero_pairs = [(s, t, v, span) for s, t, v, span in mid_pairs if v < R_ZERO_THRESH]
if zero_pairs:
    zero_pairs_sorted = sorted(zero_pairs, key=lambda x: (-x[3], x[0]))
    best_s, best_t, best_R, _ = zero_pairs_sorted[0]
elif mid_pairs:
    mid_pairs_sorted = sorted(mid_pairs, key=lambda x: (x[2], -x[3]))
    best_s, best_t, best_R, _ = mid_pairs_sorted[0]
else:
    best_s = best_t = -1; best_R = float("nan")

compress_R = R_mat[COMPRESS_SRC, COMPRESS_TGT] \
             if (COMPRESS_TGT < L and not np.isnan(R_mat[COMPRESS_SRC, COMPRESS_TGT])) \
             else float("nan")

print(f"\n{'='*70}")
print(f"  RECOUPLING RATIO SUMMARY  (FL+MDCA calibrated teacher latents)")
print(f"{'='*70}")
if zero_pairs:
    print(f"\n  All R≈0 pairs in middle layers ({len(zero_pairs)} found) — sorted by span ↓:")
    for s, t, v, span in sorted(zero_pairs, key=lambda x: (-x[3], x[0])):
        marker = "  ◀ WIDEST (selected)" if (s == best_s and t == best_t) else ""
        print(f"    OA[{s}] → OA[{t}]   span={span}   R = {v:.9f}{marker}")
else:
    print(f"\n  No R≈0 pairs found in middle layers.")
print(f"\n  Best middle-layer block : OA[{best_s}] → OA[{best_t}]   R = {best_R:.9f}")
print(f"  Fixed compression block : OA[{COMPRESS_SRC}] → OA[{COMPRESS_TGT}]   R = {compress_R:.9f}")
print(f"  Scan time : {rr_time:.1f}s  |  Collect time : {collect_time:.1f}s")
print(f"{'='*70}")

# ── Heatmap ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"Cell 2 — Recoupling Ratio  (FL+MDCA calibrated teacher latents)\n"
    f"teacher test_acc={mdca_res['test_acc']:.2f}%  "
    f"test_ECE={mdca_res['test_ece']:.2f}%  test_SCE={mdca_res['test_sce']:.4f}e-3",
    fontweight="bold", fontsize=9)
masked = np.where(np.isnan(R_mat), -0.05, R_mat)
im = axes[0].imshow(masked, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
plt.colorbar(im, ax=axes[0], label="R  (0 = compressible, 1 = avoid)")
axes[0].set(xlabel="Target OA Layer", ylabel="Source OA Layer",
            title="Recoupling Ratio R[src → tgt]\n(LFT paper Eq 6-7, exact EMD)")
axes[0].set_xticks(range(L)); axes[0].set_yticks(range(L))
axes[0].set_xticklabels([f"OA{i}" for i in range(L)], fontsize=7)
axes[0].set_yticklabels([f"OA{i}" for i in range(L)], fontsize=7)
for s in range(L):
    for t in range(L):
        v = R_mat[s, t]
        if not np.isnan(v):
            axes[0].text(t, s, f"{v:.3f}", ha="center", va="center",
                         fontsize=7 if L > 8 else 9, fontweight="bold",
                         color="white" if v > 0.5 else "black")
if best_s >= 0:
    axes[0].scatter([best_t], [best_s], marker="*", s=350, c="blue", zorder=7,
                    label=f"Best middle: OA{best_s}→OA{best_t}  R={best_R:.3f}")
if COMPRESS_TGT < L:
    axes[0].scatter([COMPRESS_TGT], [COMPRESS_SRC], marker="o", s=250,
                    facecolors="none", edgecolors="magenta", linewidths=2.5, zorder=8,
                    label=f"Compression: OA{COMPRESS_SRC}→OA{COMPRESS_TGT}  R={compress_R:.3f}")
axes[0].legend(fontsize=8, loc="upper left")

row_mins = []
for s in range(L):
    vals = [R_mat[s, t] for t in range(s+1, L) if not np.isnan(R_mat[s, t])]
    row_mins.append(min(vals) if vals else np.nan)
bar_colors = ["lightgray" if np.isnan(v) else "#27ae60" if v < 0.10 else
              "#2ecc71" if v < 0.30 else "#f39c12" if v < 0.60 else "#e74c3c" for v in row_mins]
bar_vals = [0.0 if np.isnan(v) else v for v in row_mins]
bars = axes[1].bar(range(L), bar_vals, color=bar_colors, edgecolor="k", linewidth=0.8)
axes[1].axhline(0.10, color="#27ae60", ls="--", lw=1.5, label="R=0.10 highly compressible")
axes[1].axhline(0.30, color="#f39c12", ls="--", lw=1.5, label="R=0.30 moderate")
axes[1].axhline(0.60, color="#e74c3c", ls="--", lw=1.5, label="R=0.60 avoid")
for bar, val in zip(bars, bar_vals):
    if val > 0:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
axes[1].set(xlabel="Source OA Layer", ylabel="Min R to any later layer",
            title="Best Compressibility per Source Layer", ylim=[0, 1.15])
axes[1].set_xticks(range(L)); axes[1].set_xticklabels([f"OA{i}" for i in range(L)], fontsize=7)
axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("cell2_rr_heatmap.png", dpi=150, bbox_inches="tight")
plt.show(); print("Saved → cell2_rr_heatmap.png")

# ── Save ──────────────────────────────────────────────────────────────────────
torch.save({
    "R_mat": R_mat,
    "lft_src": COMPRESS_SRC, "lft_tgt": COMPRESS_TGT, "compress_R": compress_R,
    "x0": latents_in[COMPRESS_SRC],
    "x1": latents_in[COMPRESS_TGT] if COMPRESS_TGT < L else latents_in[-1],
    "mid_best_s": best_s, "mid_best_t": best_t, "mid_best_R": best_R,
    "mid_zero_pairs": [(s, t, float(v)) for s, t, v, _ in zero_pairs],
    "cfg": CFG,
    "teacher_source": "FL+MDCA",
    "teacher_test_acc": mdca_res["test_acc"],
    "teacher_test_ece": mdca_res["test_ece"],
    "teacher_test_sce": mdca_res["test_sce"],
    "rr_time_s": rr_time, "collect_time_s": collect_time,
    "O_M": OT_CLOUDS, "n_clouds": N_clouds, "latent_dim": D_feat, "n_tokens": N_tok,
}, "rr_results.pt")
print("Saved → rr_results.pt")
print(f"  teacher_source  : FL+MDCA  (pct_mdca.pth)")
print(f"  lft_src/lft_tgt : {COMPRESS_SRC}/{COMPRESS_TGT}")
print(f"  mid_best        : OA[{best_s}]→OA[{best_t}]  R={best_R:.9f}")
print("\nCell 2 complete.")