# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 3A — LFT-PCT  Block OA[3]->OA[9]  PHASE 1 (FW velocity training)    ║
# ║  Teacher  : pct_mdca.pth  (FL+MDCA calibrated teacher)                     ║
# ║  Latents  : collected from FL+MDCA teacher (well-calibrated representations)║
# ║  Requires : pct_mdca.pth | pct_compare_stats.pt | rr_results.pt            ║
# ║             flop_utils.py                                                   ║
# ║  Saves    : lft_blockB_vel.pth | cell3a_results.pt                         ║
# ║  NOTE     : pytorch3d replaced with pure-PyTorch FPS + KNN                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import sys, subprocess, os, glob, time, warnings
warnings.filterwarnings("ignore")

def pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

pip("numpy", "matplotlib", "h5py", "tqdm", "scipy")

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {DEVICE}")
torch.manual_seed(42); np.random.seed(42)

for f in ["pct_mdca.pth", "pct_compare_stats.pt", "rr_results.pt", "flop_utils.py"]:
    assert os.path.exists(f), f"Missing {f} — run comparison cell and Cell 2 first."

from flop_utils import flops_pct_teacher, flops_lft_pct, human, flops_offset_attention

# ── Load stats from FL+MDCA comparison results ────────────────────────────────
compare_stats = torch.load("pct_compare_stats.pt", weights_only=False)
CFG           = compare_stats["cfg"]
mdca_res      = compare_stats["results"]["FL+MDCA"]
rr            = torch.load("rr_results.pt", weights_only=False)

# ── Override resource-constrained parameters ──────────────────────────────────
CFG["batch_size"]   = 32
CFG["num_workers"]  = 2

print(f"FL+MDCA teacher | test_acc={mdca_res['test_acc']:.2f}% | "
      f"test_ECE={mdca_res['test_ece']:.2f}% | test_SCE={mdca_res['test_sce']:.4f}e-3")
print(f"Architecture    | n_attn={CFG['n_attn']}  embed_dim={CFG['embed_dim']}")
print(f"RR latents from : {rr.get('teacher_source', 'FL+MDCA')} teacher")

# ── Block definition ───────────────────────────────────────────────────────────
BLOCK      = dict(src=3, tgt=9, name="BlockB_OA3to9")
N_REPLACED = BLOCK["tgt"] - BLOCK["src"]

print(f"\nBlock     : OA[{BLOCK['src']}..{BLOCK['tgt']-1}]  ({N_REPLACED} layers -> 1 FW net)")
print(f"Pre-flow  : OA[0..{BLOCK['src']-1}]  ({BLOCK['src']} layers)")
print(f"Post-flow : OA[{BLOCK['tgt']}..{CFG['n_attn']-1}]  ({CFG['n_attn']-BLOCK['tgt']} layers)")

VEL_EPOCHS    = 250
VEL_LR        = 3e-4
FW_K          = 3
BATCH_SIZE    = CFG["batch_size"]
NUM_WORKERS   = CFG["num_workers"]
VEL_PATIENCE  = 35   # scaled up proportionally with epoch budget
VEL_MIN_DELTA = 1e-5

# ══════════════════════════════════════════════════════════════════════════════
#  PURE-PYTORCH REPLACEMENTS FOR pytorch3d
#  - sample_farthest_points : greedy iterative FPS
#  - knn_points              : batched cdist-based KNN (returns .idx attribute)
# ══════════════════════════════════════════════════════════════════════════════

def sample_farthest_points_pt(xyz, K):
    """
    Pure-PyTorch Farthest Point Sampling.
    xyz : (B, N, 3)
    Returns (sampled_xyz, indices)  shapes (B, K, 3), (B, K)
    """
    B, N, _ = xyz.shape
    K = min(K, N)
    device = xyz.device
    indices  = torch.zeros(B, K, dtype=torch.long, device=device)
    # start from point 0
    cur      = torch.zeros(B, dtype=torch.long, device=device)
    dists    = torch.full((B, N), float('inf'), device=device)

    for i in range(K):
        indices[:, i] = cur
        # (B, 3)
        cur_pts = xyz[torch.arange(B, device=device), cur, :]   # (B, 3)
        # squared distances from cur to all points
        d = ((xyz - cur_pts.unsqueeze(1)) ** 2).sum(-1)         # (B, N)
        dists = torch.minimum(dists, d)
        cur   = dists.argmax(dim=1)                              # (B,)

    sampled = xyz[torch.arange(B, device=device).unsqueeze(1),
                  indices, :]                                    # (B, K, 3)
    return sampled, indices


class _KNNResult:
    """Thin wrapper so knn_points_pt returns an object with a .idx attribute."""
    def __init__(self, idx):
        self.idx = idx


def knn_points_pt(query, ref, K):
    """
    Batched KNN using torch.cdist.
    query : (B, M, 3)
    ref   : (B, N, 3)
    Returns object with .idx of shape (B, M, K)
    """
    K = min(K, ref.shape[1])
    dists = torch.cdist(query, ref)          # (B, M, N)
    idx   = dists.topk(K, dim=-1, largest=False).indices   # (B, M, K)
    return _KNNResult(idx)


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR    = "modelnet40_ply_hdf5_2048"
TRAIN_FILES = sorted(glob.glob(os.path.join(DATA_DIR, "ply_data_train*.h5")))
TEST_FILES  = sorted(glob.glob(os.path.join(DATA_DIR, "ply_data_test*.h5")))
assert TRAIN_FILES, "ModelNet40 not found."

def pc_normalize(pc):
    pc = pc - pc.mean(0)
    return pc / (np.sqrt((pc**2).sum(1)).max() + 1e-8)

class MN40(Dataset):
    def __init__(self, files, n_pts=1024, augment=False, in_dropout=0.0, seed_offset=0):
        self.n = n_pts; self.augment = augment
        self.drop = in_dropout; self.seed_offset = seed_offset
        d, l = [], []
        for f in files:
            with h5py.File(f) as h:
                d.append(h["data"][:].astype(np.float32))
                l.append(h["label"][:].flatten().astype(np.int64))
        self.data   = np.concatenate(d)
        self.labels = np.concatenate(l)
        print(f"  Loaded {len(self.data):,} clouds  augment={augment}")
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        pts = self.data[i].copy()
        rng = np.random.RandomState(i + self.seed_offset)
        idx = rng.choice(pts.shape[0], self.n, replace=False)
        pts = pc_normalize(pts[idx])
        if self.augment:
            pts += np.random.uniform(-0.2, 0.2, (1, 3)).astype(np.float32)
            pts *= np.random.uniform(0.67, 1.5,  (1, 3)).astype(np.float32)
            n = pts.shape[0]; dd = int(n * self.drop * np.random.rand())
            if dd > 0:
                di = np.random.choice(n, dd, replace=False)
                pts[di] = pts[np.random.choice(n, dd, replace=True)]
        return torch.from_numpy(pts), int(self.labels[i])

def strat_split(labels, ratio=0.20, seed=42):
    rng = np.random.RandomState(seed); tr, va = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]; rng.shuffle(idx)
        nv  = max(1, int(len(idx) * ratio))
        va += idx[:nv].tolist(); tr += idx[nv:].tolist()
    return np.array(tr), np.array(va)

class SubsetDS(Dataset):
    def __init__(self, base, indices, augment, in_dropout=0.0):
        self.base = base; self.idx = indices
        self.augment = augment; self.drop = in_dropout
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        pts, label = self.base[self.idx[i]]
        if isinstance(pts, torch.Tensor): pts = pts.numpy()
        pts = pts.copy().astype(np.float32)
        if self.augment:
            pts += np.random.uniform(-0.2, 0.2, (1, 3)).astype(np.float32)
            pts *= np.random.uniform(0.67, 1.5,  (1, 3)).astype(np.float32)
            n = pts.shape[0]; dd = int(n * self.drop * np.random.rand())
            if dd > 0:
                di = np.random.choice(n, dd, replace=False)
                pts[di] = pts[np.random.choice(n, dd, replace=True)]
        return torch.from_numpy(pts), label

print("\nBuilding datasets ...")
train_pool = MN40(TRAIN_FILES, CFG["n_pts"], augment=False, seed_offset=0)
test_pool  = MN40(TEST_FILES,  CFG["n_pts"], augment=False, seed_offset=99999)
tr_idx, va_idx = strat_split(train_pool.labels, CFG.get("val_ratio", 0.2), CFG.get("seed", 42))
train_ds = SubsetDS(train_pool, tr_idx, augment=True, in_dropout=CFG.get("in_dropout", 0.0))
val_ds   = SubsetDS(train_pool, va_idx, augment=False)

# persistent_workers=False with 2 workers avoids zombie processes on Linux
DLK = dict(num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
           persistent_workers=False)
train_dl = DataLoader(train_ds,  BATCH_SIZE, shuffle=True,  drop_last=True,  **DLK)
val_dl   = DataLoader(val_ds,    BATCH_SIZE, shuffle=False, **DLK)
test_dl  = DataLoader(test_pool, BATCH_SIZE, shuffle=False, **DLK)
print(f"Train:{len(tr_idx):,}  Val:{len(va_idx):,}  Test:{len(test_pool):,}")

# ══════════════════════════════════════════════════════════════════════════════
#  PCT BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════════════
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
        B, N, _ = xyz.shape
        no = min(self.nout, N)
        k  = min(self.k,    N)

        # ── FPS (pure PyTorch) ────────────────────────────────────────────────
        sxyz, fi = sample_farthest_points_pt(xyz, K=no)   # (B,no,3), (B,no)

        # ── KNN (pure PyTorch) ───────────────────────────────────────────────
        ki = knn_points_pt(sxyz, xyz, K=k).idx            # (B, no, k)

        C   = feat.shape[-1]
        nbr = feat.unsqueeze(1).expand(-1, no, -1, -1).gather(
              2, ki.unsqueeze(-1).expand(-1, -1, -1, C))
        ctr = feat.gather(1, fi.unsqueeze(-1).expand(-1, -1, C))
        agg = torch.cat(
              [nbr - ctr.unsqueeze(2).expand_as(nbr),
               ctr.unsqueeze(2).expand_as(nbr)], -1)
        Bo, No, K2, C2 = agg.shape
        out = self.l2(self.l1(agg.reshape(Bo * No * K2, C2)))
        return sxyz, out.reshape(Bo, No, K2, -1).max(2).values

class NE(nn.Module):
    def __init__(self, ns1, ns2, k, D):
        super().__init__()
        self.lbr = LBR(3, 64)
        self.sg1 = SGModule(ns1, k, 64, 64)
        self.sg2 = SGModule(ns2, k, 64, D)
    def forward(self, xyz):
        f = self.lbr(xyz)
        x2, f2 = self.sg1(xyz, f)
        x3, f3 = self.sg2(x2, f2)
        return x3, f3

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
            nn.Linear(256,  256, bias=False), nn.BatchNorm1d(256), nn.ReLU(True), nn.Dropout(cfg["drop_rate"]),
            nn.Linear(256, cfg["n_classes"]),
        )
    def forward(self, xyz):
        _, h = self.ne(xyz); outs = []
        for oa in self.oas: h = oa(h); outs.append(h)
        f = self.proj(torch.cat(outs, -1))
        g = torch.cat([f.max(1).values, f.mean(1)], -1)
        return self.head(g)

def nparams(m):     return sum(p.numel() for p in m.parameters())
def n_trainable(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

# ══════════════════════════════════════════════════════════════════════════════
#  FW VELOCITY ESTIMATOR
# ══════════════════════════════════════════════════════════════════════════════
class FWVelocityEstimator(nn.Module):
    def __init__(self, D, da):
        super().__init__()
        self.D = D
        self.t_mlp = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2 * D))
        self.norm = nn.LayerNorm(D)
        self.oa   = OA(D, da)
    def forward(self, xt, t):
        t_in  = t.unsqueeze(-1).float()
        out   = self.t_mlp(t_in)
        scale, shift = out.chunk(2, dim=-1)
        h = self.norm(xt) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        h = self.oa(h)
        return h - xt
    def step(self, xt, t_cur, t_nxt):
        d   = (t_nxt - t_cur).view(-1, 1, 1)
        t_m = (t_cur + t_nxt) / 2
        v1  = self.forward(xt, t_cur)
        v2  = self.forward(xt + 0.5 * d * v1, t_m)
        return xt + d * v2
    def transport(self, x0, k=3):
        x = x0; B = x0.shape[0]
        for i in range(k):
            t_c = torch.full((B,), i / k,     device=x.device, dtype=x.dtype)
            t_n = torch.full((B,), (i+1) / k, device=x.device, dtype=x.dtype)
            x   = self.step(x, t_c, t_n)
        return x

def flow_walking_loss(vel_net, x0, x1, k=3):
    B = x0.shape[0]
    t_inner  = torch.rand(k - 1).sort().values.tolist()
    ts_float = [0.0] + t_inner + [1.0]
    x_cur = x0
    for i in range(k):
        t_c = torch.full((B,), ts_float[i],   device=x0.device, dtype=x0.dtype)
        t_n = torch.full((B,), ts_float[i+1], device=x0.device, dtype=x0.dtype)
        x_cur = vel_net.step(x_cur, t_c, t_n)
    return F.mse_loss(x_cur, x1)

# ══════════════════════════════════════════════════════════════════════════════
#  LFT-PCT
# ══════════════════════════════════════════════════════════════════════════════
class LFT_PCT(nn.Module):
    def __init__(self, cfg, fw_net, src, tgt, fw_k=3):
        super().__init__()
        D = cfg["embed_dim"]; da = cfg["da"]; L = cfg["n_attn"]
        self.src, self.tgt, self.fw_k, self.L = src, tgt, fw_k, L
        self.kept    = [i for i in range(L) if not (src <= i < tgt)]
        self.n_slots = len(self.kept) + 1
        self.ne   = NE(cfg["n_sg1"], cfg["n_sg2"], cfg["k_sg"], D)
        self.oas  = nn.ModuleList([OA(D, da) for _ in self.kept])
        self.fw   = fw_net
        self.proj = nn.Linear(D * self.n_slots, 1024, bias=False)
        self.head = nn.Sequential(
            nn.Linear(2048, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(True), nn.Dropout(cfg["drop_rate"]),
            nn.Linear(256,  256, bias=False), nn.BatchNorm1d(256), nn.ReLU(True), nn.Dropout(cfg["drop_rate"]),
            nn.Linear(256, cfg["n_classes"]),
        )
    def forward(self, xyz):
        _, h = self.ne(xyz); outs = []; oa_i = 0; B = h.shape[0]
        for i in range(self.L):
            if i == self.src:
                if not torch.is_grad_enabled():
                    h = self.fw.transport(h, k=self.fw_k)
                else:
                    for s in range(self.fw_k):
                        t_c = torch.full((B,), s/self.fw_k,     device=h.device)
                        t_n = torch.full((B,), (s+1)/self.fw_k, device=h.device)
                        h = self.fw.step(h, t_c, t_n)
                outs.append(h)
            elif self.src < i < self.tgt:
                pass
            else:
                h = self.oas[oa_i](h); outs.append(h); oa_i += 1
        f = self.proj(torch.cat(outs, -1))
        g = torch.cat([f.max(1).values, f.mean(1)], -1)
        return self.head(g)

def transplant_weights(lft, teacher_sd, kept, src, tgt, D, L):
    l_sd = lft.state_dict()
    for k in list(l_sd.keys()):
        if k.startswith("ne."):   l_sd[k] = teacher_sd[k]
        if k.startswith("head."): l_sd[k] = teacher_sd[k]
    for new_i, old_i in enumerate(kept):
        pn, po = f"oas.{new_i}.", f"oas.{old_i}."
        for k in list(l_sd.keys()):
            if k.startswith(pn): l_sd[k] = teacher_sd[po + k[len(pn):]]
    lft.load_state_dict(l_sd)
    t_proj = teacher_sd["proj.weight"]
    blocks = [t_proj[:, i*D:(i+1)*D] for i in range(L)]
    flow_block = sum(blocks[i] for i in range(src, tgt))
    new_proj_w = torch.cat(
        [blocks[i] for i in range(0, src)] + [flow_block] + [blocks[i] for i in range(tgt, L)],
        dim=1)
    lft.proj.weight.data.copy_(new_proj_w)
    print(f"  Transplanted: NE, {len(kept)} kept OA layers, head, proj "
          f"(flow slot = sum of {tgt-src} teacher blocks)")

@torch.no_grad()
def capture_latents(teacher, src, tgt, xyz):
    _, h = teacher.ne(xyz)
    for i in range(src): h = teacher.oas[i](h)
    x0 = h.clone()
    for i in range(src, tgt): h = teacher.oas[i](h)
    return x0, h

@torch.no_grad()
def evaluate(model, dl, label=""):
    model.eval(); ok = tot = 0
    for pts, lbl in dl:
        pts = pts.to(DEVICE, non_blocking=True); lbl = lbl.to(DEVICE, non_blocking=True)
        with autocast(DEVICE):
            ok += (model(pts).argmax(1) == lbl).sum().item()
        tot += lbl.size(0)
    acc = 100.0 * ok / tot
    if label: print(f"  [{label}]  {acc:.2f}%")
    return acc, tot

# ══════════════════════════════════════════════════════════════════════════════
#  BUILD MODELS
# ══════════════════════════════════════════════════════════════════════════════
D  = CFG["embed_dim"]; da = CFG["da"]; L = CFG["n_attn"]

# Load FL+MDCA calibrated teacher as frozen reference
teacher = PCT(CFG).to(DEVICE)
teacher.load_state_dict(torch.load("pct_mdca.pth", map_location=DEVICE, weights_only=True))
teacher.eval()
for p in teacher.parameters(): p.requires_grad_(False)
n_pct = nparams(teacher)
print(f"\nFL+MDCA teacher loaded | params={n_pct:,} | "
      f"test_ECE={mdca_res['test_ece']:.2f}%  test_SCE={mdca_res['test_sce']:.4f}e-3")

fw    = FWVelocityEstimator(D, da).to(DEVICE)
model = LFT_PCT(CFG, fw, BLOCK["src"], BLOCK["tgt"], fw_k=FW_K).to(DEVICE)
transplant_weights(model, teacher.state_dict(), model.kept, BLOCK["src"], BLOCK["tgt"], D, L)

pct_flops = flops_pct_teacher(CFG)
lft_flops = flops_lft_pct(CFG, BLOCK, fw_k=FW_K)
flop_saving = pct_flops['total'] - lft_flops['total']
n_tok_train = pct_flops["n_tok"]

print(f"\n  FLOPs/sample — PCT teacher : {human(pct_flops['total'])}")
print(f"  FLOPs/sample — LFT-PCT    : {human(lft_flops['total'])}")
print(f"  FLOP saving               : {human(flop_saving)} ({100*flop_saving/pct_flops['total']:.1f}%)")

def flops_velocity_net_call(n_tok):
    f = 2*64 + 64 + 2*64*64 + 64 + 2*64*(2*D) + 2*D
    f += 5*n_tok*D + 2*n_tok*D + flops_offset_attention(D, da, n_tok) + n_tok*D
    return f

flops_per_fw_loss_call    = (2*FW_K) * flops_velocity_net_call(n_tok_train) + FW_K * (4 * n_tok_train * D)
flops_per_capture_call    = BLOCK["tgt"] * flops_offset_attention(D, da, n_tok_train)
flops_per_train_sample_p1 = flops_per_capture_call + flops_per_fw_loss_call

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — VELOCITY TRAINING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"  PHASE 1 — FW Velocity Training  ({VEL_EPOCHS} epochs, patience={VEL_PATIENCE})")
print(f"  Teacher source: FL+MDCA calibrated PCT  (pct_mdca.pth)")
print(f"  OA[{BLOCK['src']}..{BLOCK['tgt']-1}]  ({N_REPLACED} layers -> 1 net)")
print(f"  batch_size={BATCH_SIZE}  num_workers={NUM_WORKERS}")
print("=" * 65)

for p in model.parameters(): p.requires_grad_(False)
for p in model.fw.parameters(): p.requires_grad_(True)
n_total = nparams(model)

opt_v     = torch.optim.AdamW(model.fw.parameters(), lr=VEL_LR, weight_decay=1e-4)
sched_v   = torch.optim.lr_scheduler.CosineAnnealingLR(opt_v, VEL_EPOCHS, eta_min=1e-6)
scaler_p1 = GradScaler(DEVICE)

p1_losses, p1_times, p1_nmse, p1_cum_flops = [], [], [], []
best_vel_loss  = float("inf")
best_val_nmse  = float("inf")
best_epoch     = 0
train_loss_at_best    = float("inf")
epochs_since_improve  = 0
cumulative_flops_p1   = 0
total_p1_val_flops    = 0

def full_val_nmse(transport_fn):
    global total_p1_val_flops
    total_mse = 0.0; total_sq = 0.0
    with torch.no_grad():
        for pts_v, _ in val_dl:
            pts_v = pts_v.to(DEVICE, non_blocking=True)
            xv0, xv1 = capture_latents(teacher, BLOCK["src"], BLOCK["tgt"], pts_v)
            xv0, xv1 = xv0.float(), xv1.float()
            xv_hat = transport_fn(xv0)
            total_mse += F.mse_loss(xv_hat, xv1, reduction="sum").item()
            total_sq  += xv1.pow(2).sum().item()
    total_p1_val_flops += flops_per_train_sample_p1 * len(va_idx)
    return total_mse / (total_sq + 1e-8)

identity_nmse = full_val_nmse(lambda x0: x0)
print(f"\n  Identity (skip) baseline NMSE: {identity_nmse:.6f}")

t_vel_start = time.perf_counter()
stopped_early = False

for ep in range(1, VEL_EPOCHS + 1):
    model.ne.eval(); model.oas.eval(); model.proj.eval(); model.head.eval()
    model.fw.train()
    t0 = time.perf_counter(); ep_loss = ep_n = 0

    for pts, _ in train_dl:
        pts = pts.to(DEVICE, non_blocking=True)
        opt_v.zero_grad(set_to_none=True)
        with autocast(DEVICE):
            x0, x1 = capture_latents(teacher, BLOCK["src"], BLOCK["tgt"], pts)
            x0 = x0.float(); x1 = x1.float()
            loss = flow_walking_loss(model.fw, x0, x1, k=FW_K)
        scaler_p1.scale(loss).backward()
        scaler_p1.unscale_(opt_v)
        nn.utils.clip_grad_norm_(model.fw.parameters(), 1.0)
        scaler_p1.step(opt_v); scaler_p1.update()
        ep_loss += loss.item() * pts.size(0); ep_n += pts.size(0)
        cumulative_flops_p1 += pts.size(0) * flops_per_train_sample_p1

    sched_v.step()
    ep_time  = time.perf_counter() - t0
    cum_time = time.perf_counter() - t_vel_start
    avg_loss = ep_loss / ep_n
    if avg_loss < best_vel_loss: best_vel_loss = avg_loss

    model.fw.eval()
    val_nmse = full_val_nmse(lambda x0: model.fw.transport(x0, k=FW_K))

    is_best = val_nmse < (best_val_nmse - VEL_MIN_DELTA)
    if is_best:
        best_val_nmse = val_nmse; best_epoch = ep; train_loss_at_best = avg_loss
        epochs_since_improve = 0
        torch.save(model.fw.state_dict(), "lft_blockB_vel.pth")
    else:
        epochs_since_improve += 1

    p1_losses.append(avg_loss); p1_times.append(ep_time)
    p1_nmse.append(val_nmse);   p1_cum_flops.append(cumulative_flops_p1)

    if ep % 10 == 0 or ep == 1:
        print(f"  ep{ep:3d}/{VEL_EPOCHS} | loss={avg_loss:.6f} | nmse={val_nmse:.6f} "
              f"{'*' if is_best else ' '} | {ep_time:.1f}s | {cum_time/60:.2f}m | "
              f"{human(cumulative_flops_p1)}")

    if epochs_since_improve >= VEL_PATIENCE:
        print(f"\n  Early stop @ ep {ep} (best @ ep {best_epoch})")
        stopped_early = True; break

total_vel_time   = time.perf_counter() - t_vel_start
n_vel_epochs_run = len(p1_losses)
print(f"\n  Phase 1 done | {n_vel_epochs_run} epochs{' (early-stopped)' if stopped_early else ''} | "
      f"{total_vel_time/60:.2f}min")
print(f"  Best val NMSE={best_val_nmse:.6f} @ ep {best_epoch} | "
      f"identity baseline={identity_nmse:.6f}")
print(f"  FW net {'BETTER' if best_val_nmse < identity_nmse else 'WORSE'} than identity")
print(f"  Total Phase-1 training FLOPs: {human(cumulative_flops_p1)}")

model.fw.load_state_dict(torch.load("lft_blockB_vel.pth", map_location=DEVICE, weights_only=True))

# ══════════════════════════════════════════════════════════════════════════════
#  ZERO-SHOT TEST EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  ZERO-SHOT TEST EVALUATION  (no finetuning)")
print(f"  Teacher: FL+MDCA  (test_ECE={mdca_res['test_ece']:.2f}%)")
print("=" * 65)

for p in model.parameters(): p.requires_grad_(False)
val_acc_zs,  _      = evaluate(model,   val_dl,  label="LFT zero-shot Val ")
test_acc_zs, n_test = evaluate(model,   test_dl, label="LFT zero-shot Test")
test_acc_pct, _     = evaluate(teacher, test_dl, label="FL+MDCA teacher Test")

test_flops_lft_zs = lft_flops["total"] * n_test
test_flops_pct    = pct_flops["total"] * n_test
print(f"\n  FL+MDCA teacher test acc : {test_acc_pct:.2f}%")
print(f"  LFT zero-shot test acc   : {test_acc_zs:.2f}%")
print(f"  Drop vs teacher          : {test_acc_pct - test_acc_zs:+.2f}%")
print(f"  Test FLOP saving (LFT vs PCT): "
      f"{human(test_flops_pct - test_flops_lft_zs)}  "
      f"({100*(test_flops_pct-test_flops_lft_zs)/test_flops_pct:.1f}%)")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle(
    f"Phase 1 — FW Velocity Training  |  Teacher: FL+MDCA\n"
    f"{n_vel_epochs_run} epochs{', early-stopped' if stopped_early else ''} | "
    f"Best NMSE: {best_val_nmse:.6f} | Identity: {identity_nmse:.6f} | "
    f"ZS test acc: {test_acc_zs:.2f}%  (teacher: {test_acc_pct:.2f}%)",
    fontsize=9, fontweight="bold")
axes[0].plot(p1_losses, color="steelblue", lw=2)
axes[0].set(title="FW MSE Loss (training)", xlabel="Epoch", ylabel="Loss"); axes[0].grid(alpha=0.3)
axes[1].plot(p1_nmse, color="#3498db", lw=2, label="Val NMSE")
axes[1].axhline(identity_nmse, color="gray", ls="--", lw=1.5, label="identity baseline")
best_ep_idx = int(np.argmin(p1_nmse))
axes[1].scatter([best_ep_idx], [p1_nmse[best_ep_idx]], color="#e74c3c", zorder=5,
                label=f"checkpoint (ep {best_ep_idx+1})")
axes[1].set(title="Val NMSE", xlabel="Epoch", ylabel="NMSE"); axes[1].legend(); axes[1].grid(alpha=0.3)
axes[2].plot([f/1e9 for f in p1_cum_flops], color="#8e44ad", lw=2)
axes[2].set(title="Cumulative Training FLOPs (fwd only)", xlabel="Epoch", ylabel="GFLOPs"); axes[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("cell3a_phase1_velocity.png", dpi=150, bbox_inches="tight")
plt.show(); print("Saved -> cell3a_phase1_velocity.png")

# Save
torch.save({
    "cfg": CFG, "block": BLOCK, "fw_k": FW_K,
    "vel_epochs_run": n_vel_epochs_run, "vel_epochs_max": VEL_EPOCHS,
    "stopped_early": stopped_early,
    "vel_patience": VEL_PATIENCE, "vel_min_delta": VEL_MIN_DELTA,
    "n_lft_total": n_total, "n_fw": nparams(model.fw), "n_pct": n_pct,
    "identity_nmse": identity_nmse,
    "p1_losses": p1_losses, "p1_times": p1_times, "p1_nmse": p1_nmse,
    "p1_cum_flops": p1_cum_flops,
    "total_vel_time_s": total_vel_time, "best_vel_loss": best_vel_loss,
    "best_val_nmse": best_val_nmse, "best_epoch": best_epoch,
    "val_acc_zs": val_acc_zs, "test_acc_zs": test_acc_zs,
    "test_acc_pct": test_acc_pct,
    "teacher_source": "FL+MDCA",
    "teacher_test_acc": mdca_res["test_acc"],
    "teacher_test_ece": mdca_res["test_ece"],
    "teacher_test_sce": mdca_res["test_sce"],
    "pct_flops": pct_flops, "lft_flops": lft_flops,
    "flops_per_train_sample_p1": flops_per_train_sample_p1,
    "n_test": n_test,
    "test_flops_pct": test_flops_pct,
    "test_flops_lft_zs": test_flops_lft_zs,
    # These two are needed by Cell 3B
    "total_p1_train_flops": cumulative_flops_p1,
    "total_p1_val_flops":   total_p1_val_flops,
}, "cell3a_results.pt")
print("Saved: lft_blockB_vel.pth | cell3a_results.pt | cell3a_phase1_velocity.png")
print("\nCell 3A complete. Proceed to Cell 3B for finetuning.")