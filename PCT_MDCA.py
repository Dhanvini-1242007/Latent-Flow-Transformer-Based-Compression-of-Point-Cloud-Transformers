# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CELL 1 (COMPARISON) — PCT trained with CE  vs  FL+MDCA               ║
# ║  100-250 epochs each, identical architecture/data/seed/optimizer.         ║
# ║  Compares: Test Accuracy, Test ECE, Test SCE, reliability diagrams.   ║
# ║  Saves : pct_ce.pth | pct_mdca.pth | pct_compare_stats.pt             ║
# ║  NOTE  : pytorch3d replaced with pure-PyTorch equivalents              ║
# ╚══════════════════════════════════════════════════════════════════════╝

import sys, subprocess, os, zipfile, glob, time, warnings, copy
warnings.filterwarnings("ignore")

# ── Install ───────────────────────────────────────────────────────────────────
def pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

pip("numpy", "matplotlib", "h5py", "gdown", "tqdm", "scipy")

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np, h5py, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Pure-PyTorch replacements for pytorch3d ops ───────────────────────────────

def sample_farthest_points(xyz, K):
    """
    Farthest Point Sampling — pure PyTorch, no pytorch3d.
    xyz : (B, N, 3)
    Returns (sampled_xyz, indices) matching pytorch3d's sample_farthest_points API.
      sampled_xyz : (B, K, 3)
      indices     : (B, K)   int64
    """
    B, N, _ = xyz.shape
    device = xyz.device
    K = min(K, N)

    indices  = torch.zeros(B, K, dtype=torch.long, device=device)
    dists    = torch.full((B, N), float("inf"), device=device)

    # Start from a random point per batch element
    cur_idx  = torch.zeros(B, dtype=torch.long, device=device)

    for i in range(K):
        indices[:, i] = cur_idx
        # Current selected point: (B, 1, 3)
        cur_pt = xyz[torch.arange(B, device=device), cur_idx].unsqueeze(1)
        # Squared distances to all points
        new_dists = ((xyz - cur_pt) ** 2).sum(-1)           # (B, N)
        dists     = torch.minimum(dists, new_dists)          # update nearest-selected dist
        cur_idx   = dists.argmax(dim=1)                      # pick farthest

    sampled_xyz = xyz[torch.arange(B, device=device).unsqueeze(1), indices]  # (B, K, 3)
    return sampled_xyz, indices


def knn_points(query, ref, K, return_nn=False):
    """
    K-Nearest Neighbours — pure PyTorch, chunked to avoid OOM.
    query : (B, M, 3)
    ref   : (B, N, 3)
    Returns object with .idx of shape (B, M, K).
    """
    B, M, _ = query.shape
    _, N, _ = ref.shape
    K = min(K, N)

    # Chunk along M so we never materialise the full (B, M, N) matrix at once.
    # chunk_size=64 keeps each slab at ~32*64*1024*4 ≈ 8 MB — well within budget.
    CHUNK = 64
    idx_parts = []

    r2 = (ref ** 2).sum(-1)          # (B, N)  — computed once, reused every chunk

    for start in range(0, M, CHUNK):
        end   = min(start + CHUNK, M)
        q_c   = query[:, start:end, :]                     # (B, c, 3)
        q2_c  = (q_c ** 2).sum(-1, keepdim=True)          # (B, c, 1)
        dot_c = torch.bmm(q_c, ref.transpose(1, 2))       # (B, c, N)
        dist2 = (q2_c + r2.unsqueeze(1) - 2 * dot_c).clamp(min=0)  # (B, c, N)
        _, ki = dist2.topk(K, dim=-1, largest=False, sorted=True)   # (B, c, K)
        idx_parts.append(ki)

    idx = torch.cat(idx_parts, dim=1)   # (B, M, K)

    class _KNNResult:
        pass
    result = _KNNResult()
    result.idx = idx
    return result

# ── Config ────────────────────────────────────────────────────────────────────
CFG = dict(
    n_classes=40, n_pts=1024, embed_dim=128, da=32, n_attn=12,
    k_sg=16, n_sg1=512, n_sg2=256,
    batch_size=32,
    epochs=250,
    lr=0.01, momentum=0.9,
    weight_decay=1e-4, drop_rate=0.5, in_dropout=0.2,
    val_ratio=0.20, num_workers=2, seed=42,
    focal_gamma=2.0,
    mdca_beta=10.0,
    ece_bins=15,
)
print("CFG:", CFG)

# ── Data download ─────────────────────────────────────────────────────────────
DATA_DIR = "modelnet40_ply_hdf5_2048"
if not os.path.isdir(DATA_DIR):
    import urllib.request
    url  = ("https://huggingface.co/datasets/Msun/modelnet40/resolve/main/"
            "modelnet40_ply_hdf5_2048.zip")
    dest = "modelnet40.zip"
    try:   urllib.request.urlretrieve(url, dest)
    except Exception:
        import gdown
        gdown.download("https://drive.google.com/uc?id=1DVBPVConAo7nEHK5R1FRQXGl6bvtm0El",
                       dest, quiet=False)
    with zipfile.ZipFile(dest) as z: z.extractall(".")
    os.remove(dest)

TRAIN_FILES = sorted(glob.glob(os.path.join(DATA_DIR, "ply_data_train*.h5")))
TEST_FILES  = sorted(glob.glob(os.path.join(DATA_DIR, "ply_data_test*.h5")))
assert TRAIN_FILES, "Data download failed."
print(f"Train HDF5: {len(TRAIN_FILES)}  Test HDF5: {len(TEST_FILES)}")

# ── Dataset ───────────────────────────────────────────────────────────────────
def pc_normalize(pc):
    pc -= pc.mean(0)
    return pc / (np.sqrt((pc**2).sum(1)).max() + 1e-8)

class MN40(Dataset):
    def __init__(self, files, n_pts=1024):
        self.n_pts = n_pts
        data, labels = [], []
        for f in files:
            with h5py.File(f) as h:
                data.append(h["data"][:].astype(np.float32))
                labels.append(h["label"][:].flatten().astype(np.int64))
        self.data   = np.concatenate(data)
        self.labels = np.concatenate(labels)
        print(f"  Loaded {len(self.data):,} clouds")
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        pts = self.data[i].copy()
        pts = pc_normalize(pts[np.random.choice(pts.shape[0], self.n_pts, replace=False)])
        return torch.from_numpy(pts), int(self.labels[i])

def strat_split(labels, ratio=0.2, seed=42):
    rng = np.random.RandomState(seed); tr, va = [], []
    for c in np.unique(labels):
        idx = np.where(labels==c)[0]; rng.shuffle(idx)
        nv = max(1, int(len(idx)*ratio))
        va += idx[:nv].tolist(); tr += idx[nv:].tolist()
    return np.array(tr), np.array(va)

print("\nBuilding datasets...")
train_pool = MN40(TRAIN_FILES, CFG["n_pts"])
test_pool  = MN40(TEST_FILES,  CFG["n_pts"])
tr_idx, va_idx = strat_split(train_pool.labels, CFG["val_ratio"], CFG["seed"])

class Subset(Dataset):
    def __init__(self, base, idx, aug, drop):
        self.base=base; self.idx=idx; self.aug=aug; self.drop=drop
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        pts, label = self.base[self.idx[i]]
        if isinstance(pts, torch.Tensor): pts=pts.numpy()
        pts = pts.copy().astype(np.float32)
        if self.aug:
            pts += np.random.uniform(-0.2,0.2,(1,3)).astype(np.float32)
            pts *= np.random.uniform(0.67,1.5,(1,3)).astype(np.float32)
            n=pts.shape[0]; d=int(n*self.drop*np.random.rand())
            if d>0:
                di=np.random.choice(n,d,replace=False)
                pts[di]=pts[np.random.choice(n,d,replace=True)]
        return torch.from_numpy(pts), label

DLK = dict(num_workers=CFG["num_workers"], pin_memory=True, persistent_workers=True)

def make_loaders():
    train_dl = DataLoader(Subset(train_pool, tr_idx, True,  CFG["in_dropout"]),
                          CFG["batch_size"], shuffle=True,  drop_last=True, **DLK)
    val_dl   = DataLoader(Subset(train_pool, va_idx, False, 0),
                          CFG["batch_size"], shuffle=False, **DLK)
    test_dl  = DataLoader(test_pool, CFG["batch_size"], shuffle=False, **DLK)
    return train_dl, val_dl, test_dl

print(f"Train:{len(tr_idx):,}  Val:{len(va_idx):,}  Test:{len(test_pool):,}")

# ── Model ─────────────────────────────────────────────────────────────────────
class LBR(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(i,o,bias=False), nn.BatchNorm1d(o), nn.ReLU(True))
    def forward(self, x):
        s=x.shape; return self.net(x.reshape(-1,s[-1])).reshape(*s[:-1],-1)

class SGModule(nn.Module):
    def __init__(self, nout, k, din, dout):
        super().__init__()
        self.nout=nout; self.k=k
        self.l1=LBR(din*2,dout); self.l2=LBR(dout,dout)
    def forward(self, xyz, feat):
        B,N,_=xyz.shape; no=min(self.nout,N); k=min(self.k,N)
        # ── replaced pytorch3d calls with pure-PyTorch equivalents ──
        sxyz, fi = sample_farthest_points(xyz, K=no)
        ki = knn_points(sxyz, xyz, K=k, return_nn=False).idx
        # ────────────────────────────────────────────────────────────
        C  = feat.shape[-1]
        nbr = feat.unsqueeze(1).expand(-1,no,-1,-1).gather(
                2, ki.unsqueeze(-1).expand(-1,-1,-1,C))
        ctr = feat.gather(1, fi.unsqueeze(-1).expand(-1,-1,C))
        ctr_e = ctr.unsqueeze(2).expand_as(nbr)
        agg = torch.cat([nbr-ctr_e, ctr_e], -1)
        Bo,No,K,C2 = agg.shape
        out = self.l2(self.l1(agg.reshape(Bo*No*K,C2)))
        return sxyz, out.reshape(Bo,No,K,-1).max(2).values

class NE(nn.Module):
    def __init__(self, ns1, ns2, k, D):
        super().__init__()
        self.lbr = LBR(3,64)
        self.sg1 = SGModule(ns1,k,64, 64)
        self.sg2 = SGModule(ns2,k,64,  D)
    def forward(self, xyz):
        f=self.lbr(xyz); x2,f2=self.sg1(xyz,f); x3,f3=self.sg2(x2,f2); return x3,f3

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
        D=cfg["embed_dim"]; da=cfg["da"]; L=cfg["n_attn"]
        self.ne   = NE(cfg["n_sg1"],cfg["n_sg2"],cfg["k_sg"],D)
        self.oas  = nn.ModuleList([OA(D,da) for _ in range(L)])
        self.proj = nn.Linear(D*L, 1024, bias=False)
        self.head = nn.Sequential(
            nn.Linear(2048,256,bias=False), nn.BatchNorm1d(256), nn.ReLU(True),
            nn.Dropout(cfg["drop_rate"]),
            nn.Linear(256, 256,bias=False), nn.BatchNorm1d(256), nn.ReLU(True),
            nn.Dropout(cfg["drop_rate"]),
            nn.Linear(256, cfg["n_classes"]),
        )
    def _encode(self, xyz):
        _,h = self.ne(xyz); outs=[]
        for oa in self.oas: h=oa(h); outs.append(h)
        return outs
    def forward(self, xyz):
        outs = self._encode(xyz)
        f    = self.proj(torch.cat(outs,-1))
        g    = torch.cat([f.max(1).values, f.mean(1)], -1)
        return self.head(g)

def nparams(m): return sum(p.numel() for p in m.parameters())

# ══════════════════════════════════════════════════════════════════════════════
#  LOSSES
# ══════════════════════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=-1)
        ce   = F.nll_loss(logp, target, reduction="none")
        pt   = logp.gather(1, target.unsqueeze(1)).squeeze(1).exp()
        return (((1 - pt) ** self.gamma) * ce).mean()

class MDCALoss(nn.Module):
    """Eq. 6: L_MDCA = (1/K) * sum_j | mean_i s_i[j] - mean_i q_i[j] |  (per mini-batch)."""
    def __init__(self, n_classes):
        super().__init__()
        self.K = n_classes
    def forward(self, logits, target):
        probs = F.softmax(logits, dim=-1)
        avg_conf  = probs.mean(dim=0)
        avg_count = F.one_hot(target, num_classes=self.K).float().mean(dim=0)
        return torch.abs(avg_conf - avg_count).mean()

class FocalMDCALoss(nn.Module):
    """L_total = L_FL + beta * L_MDCA  (paper Eq. 7, FL+MDCA best variant)."""
    def __init__(self, n_classes, gamma=2.0, beta=10.0):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma)
        self.mdca  = MDCALoss(n_classes)
        self.beta  = beta
    def forward(self, logits, target):
        l_fl, l_mdca = self.focal(logits, target), self.mdca(logits, target)
        return l_fl + self.beta * l_mdca, l_fl.detach(), l_mdca.detach()

class CEWrapper(nn.Module):
    """Wraps CrossEntropyLoss to match the (loss, component1, component2) signature
    used by FocalMDCALoss, so both losses share one training loop."""
    def __init__(self, label_smoothing=0.2):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    def forward(self, logits, target):
        l = self.ce(logits, target)
        zero = torch.zeros((), device=logits.device)
        return l, l.detach(), zero

# ══════════════════════════════════════════════════════════════════════════════
#  CALIBRATION METRICS:  ECE (Eq.4) and SCE (Eq.5)
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def compute_ece(confidences, predictions, labels, n_bins=15):
    confidences = confidences.cpu().numpy(); predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    edges = np.linspace(0, 1, n_bins + 1)
    N = len(labels); ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0: continue
        acc_bin  = (predictions[mask] == labels[mask]).mean()
        conf_bin = confidences[mask].mean()
        ece += (mask.sum() / N) * abs(acc_bin - conf_bin)
    return 100 * ece

@torch.no_grad()
def compute_sce(all_probs, labels, n_bins=15):
    """Static Calibration Error, paper Eq. 5. all_probs: (N,K) full softmax vectors."""
    all_probs = all_probs.cpu().numpy(); labels = labels.cpu().numpy()
    N, K = all_probs.shape
    edges = np.linspace(0, 1, n_bins + 1)
    sce = 0.0
    for j in range(K):
        conf_j = all_probs[:, j]
        is_j   = (labels == j).astype(np.float32)
        for i in range(n_bins):
            lo, hi = edges[i], edges[i+1]
            mask = (conf_j > lo) & (conf_j <= hi) if i > 0 else (conf_j >= lo) & (conf_j <= hi)
            if mask.sum() == 0: continue
            acc_ij  = is_j[mask].mean()
            conf_ij = conf_j[mask].mean()
            sce += (mask.sum() / N) * abs(acc_ij - conf_ij)
    return 1000 * (sce / K)

@torch.no_grad()
def evaluate(model, dl, n_bins=15, return_probs=False):
    model.eval(); ok=tot=0
    all_conf, all_pred, all_lbl, all_probs = [], [], [], []
    for pts,lbl in dl:
        pts,lbl=pts.to(DEVICE),lbl.to(DEVICE)
        with autocast(DEVICE):
            logits = model(pts)
        probs = F.softmax(logits.float(), dim=-1)
        conf, pred = probs.max(dim=-1)
        ok += (pred==lbl).sum().item(); tot += lbl.size(0)
        all_conf.append(conf); all_pred.append(pred); all_lbl.append(lbl); all_probs.append(probs)
    acc = 100*ok/tot
    conf_t, pred_t, lbl_t, probs_t = (torch.cat(all_conf), torch.cat(all_pred),
                                       torch.cat(all_lbl), torch.cat(all_probs))
    ece = compute_ece(conf_t, pred_t, lbl_t, n_bins)
    sce = compute_sce(probs_t, lbl_t, n_bins)
    if return_probs:
        return acc, ece, sce, conf_t.cpu().numpy(), pred_t.cpu().numpy(), lbl_t.cpu().numpy()
    return acc, ece, sce

# ══════════════════════════════════════════════════════════════════════════════
#  SHARED TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
def run_training(loss_name, criterion, ckpt_path):
    torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"])
    model  = PCT(CFG).to(DEVICE)
    opt    = torch.optim.SGD(model.parameters(), lr=CFG["lr"],
                              momentum=CFG["momentum"], weight_decay=CFG["weight_decay"])
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, CFG["epochs"], eta_min=1e-5)
    scaler = GradScaler(DEVICE)
    train_dl, val_dl, test_dl = make_loaders()

    hist = dict(tr_loss=[], tr_acc=[], val_acc=[], val_ece=[], val_sce=[], ep_time=[])
    best_val = 0.0

    print(f"\n{'='*65}\n  Training PCT  |  Loss = {loss_name}  |  {CFG['epochs']} epochs\n{'='*65}")
    t_wall = time.perf_counter()
    for ep in range(1, CFG["epochs"]+1):
        model.train(); t0=time.perf_counter()
        ep_loss=ep_ok=ep_n=0
        for pts,lbl in train_dl:
            pts,lbl = pts.to(DEVICE,non_blocking=True), lbl.to(DEVICE,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(DEVICE):
                logits = model(pts)
                loss, _, _ = criterion(logits, lbl)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            with torch.no_grad():
                ep_ok += (logits.argmax(1)==lbl).sum().item()
            ep_loss += loss.item()*lbl.size(0); ep_n += lbl.size(0)
        sched.step()

        val_acc, val_ece, val_sce = evaluate(model, val_dl, CFG["ece_bins"])
        ep_time = time.perf_counter()-t0
        tr_loss, tr_acc = ep_loss/ep_n, 100*ep_ok/ep_n

        hist["tr_loss"].append(tr_loss); hist["tr_acc"].append(tr_acc)
        hist["val_acc"].append(val_acc); hist["val_ece"].append(val_ece)
        hist["val_sce"].append(val_sce); hist["ep_time"].append(ep_time)

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), ckpt_path)

        if ep % 5 == 0 or ep == 1:
            print(f"[{loss_name}] ep{ep:3d}/{CFG['epochs']} | loss={tr_loss:.4f} | "
                  f"tr={tr_acc:.1f}% | val={val_acc:.1f}% (best={best_val:.1f}%) | "
                  f"val_ECE={val_ece:.2f}% | val_SCE={val_sce:.2f}e-3 | time={ep_time:.1f}s")

    total_time = time.perf_counter()-t_wall
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    test_acc, test_ece, test_sce, conf, pred, lbl = evaluate(model, test_dl, CFG["ece_bins"], return_probs=True)
    print(f"  >>> {loss_name} DONE | test_acc={test_acc:.2f}% | test_ECE={test_ece:.2f}% | "
          f"test_SCE={test_sce:.2f}e-3 | total_time={total_time/60:.1f}min")

    return dict(loss_name=loss_name, hist=hist, best_val=best_val,
                test_acc=test_acc, test_ece=test_ece, test_sce=test_sce,
                total_time=total_time, test_conf=conf, test_pred=pred, test_lbl=lbl)

# ══════════════════════════════════════════════════════════════════════════════
#  RUN BOTH
# ══════════════════════════════════════════════════════════════════════════════
results = {}
results["CE"]       = run_training("CrossEntropy+LS",
                                    CEWrapper(label_smoothing=0.2),
                                    "pct_ce.pth")
results["FL+MDCA"]  = run_training("FocalLoss+MDCA",
                                    FocalMDCALoss(CFG["n_classes"], CFG["focal_gamma"], CFG["mdca_beta"]),
                                    "pct_mdca.pth")

torch.save({"cfg": CFG, "results": results}, "pct_compare_stats.pt")

# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  FINAL COMPARISON  ({CFG['epochs']} epochs each)")
print(f"{'='*70}")
print(f"  {'Method':<18}{'Test Acc%':>12}{'Test ECE%':>12}{'Test SCE(1e-3)':>16}{'Time(min)':>12}")
for k, r in results.items():
    print(f"  {r['loss_name']:<18}{r['test_acc']:>12.2f}{r['test_ece']:>12.2f}"
          f"{r['test_sce']:>16.2f}{r['total_time']/60:>12.1f}")
print(f"{'='*70}")

# ══════════════════════════════════════════════════════════════════════════════
#  PLOTS — accuracy/ECE curves + reliability diagrams side by side
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(2, 3, figsize=(18, 9))
colors = {"CE": "steelblue", "FL+MDCA": "coral"}

for k, r in results.items():
    ax[0,0].plot(r["hist"]["val_acc"], color=colors[k], label=f"{r['loss_name']} (test={r['test_acc']:.1f}%)")
    ax[0,1].plot(r["hist"]["val_ece"], color=colors[k], label=f"{r['loss_name']} (test={r['test_ece']:.1f}%)")
    ax[0,2].plot(r["hist"]["val_sce"], color=colors[k], label=f"{r['loss_name']} (test={r['test_sce']:.1f}e-3)")
ax[0,0].set(title="Val Accuracy", xlabel="Epoch", ylabel="%"); ax[0,0].legend(fontsize=8)
ax[0,1].set(title="Val ECE", xlabel="Epoch", ylabel="%"); ax[0,1].legend(fontsize=8)
ax[0,2].set(title="Val SCE", xlabel="Epoch", ylabel="x1e-3"); ax[0,2].legend(fontsize=8)

n_bins = CFG["ece_bins"]
edges = np.linspace(0,1,n_bins+1); centers = (edges[:-1]+edges[1:])/2
for idx, (k, r) in enumerate(results.items()):
    conf, pred, lbl = r["test_conf"], r["test_pred"], r["test_lbl"]
    accs = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        accs.append((pred[mask]==lbl[mask]).mean() if mask.sum()>0 else 0)
    ax[1,idx].bar(centers, accs, width=1/n_bins, color=colors[k], alpha=0.7, label="Accuracy")
    ax[1,idx].plot([0,1],[0,1], "k--", label="Perfect calib.")
    ax[1,idx].set(title=f"Reliability — {r['loss_name']}", xlabel="Confidence", ylabel="Accuracy")
    ax[1,idx].legend(fontsize=8)
ax[1,2].axis("off")
ax[1,2].text(0.1, 0.6, f"Test Acc:\n  CE: {results['CE']['test_acc']:.2f}%\n  FL+MDCA: {results['FL+MDCA']['test_acc']:.2f}%\n\n"
                        f"Test ECE:\n  CE: {results['CE']['test_ece']:.2f}%\n  FL+MDCA: {results['FL+MDCA']['test_ece']:.2f}%\n\n"
                        f"Test SCE (1e-3):\n  CE: {results['CE']['test_sce']:.2f}\n  FL+MDCA: {results['FL+MDCA']['test_sce']:.2f}",
             fontsize=11, family="monospace", va="center")

plt.tight_layout()
plt.savefig("ce_vs_mdca_comparison.png", dpi=150)
plt.show()
print("Saved: pct_ce.pth | pct_mdca.pth | pct_compare_stats.pt | ce_vs_mdca_comparison.png")