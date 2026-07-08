# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 3B — LFT-PCT  Block OA[3]->OA[9]  PHASE 3: END-TO-END FINETUNING    ║
# ║  Teacher  : pct_mdca.pth  (FL+MDCA calibrated teacher)                     ║
# ║  Loss     : FocalLoss + MDCA  (same as teacher training, Eq.7 CVPR 2022)   ║
# ║  Requires : pct_mdca.pth | pct_compare_stats.pt | rr_results.pt            ║
# ║             flop_utils.py | lft_blockB_vel.pth | cell3a_results.pt         ║
# ║  Saves    : lft_ft_blockB.pth | cell3b_results.pt                          ║
# ║  NOTE     : pytorch3d replaced with pure-PyTorch FPS + KNN ops             ║
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

for f in ["pct_mdca.pth", "pct_compare_stats.pt", "rr_results.pt", "flop_utils.py",
          "lft_blockB_vel.pth", "cell3a_results.pt"]:
    assert os.path.exists(f), f"Missing {f} — run previous cells first."

from flop_utils import (flops_pct_teacher, flops_lft_pct, human,
                         flops_offset_attention, flops_neighbor_embedding, flops_fw_step)

# ══════════════════════════════════════════════════════════════════════════════
#  PURE-PYTORCH REPLACEMENTS FOR pytorch3d ops
#  sample_farthest_points  — farthest point sampling
#  knn_points              — k-nearest neighbours (returns idx tensor)
# ══════════════════════════════════════════════════════════════════════════════

def sample_farthest_points(xyz: torch.Tensor, K: int):
    """
    Farthest-point sampling (pure PyTorch, no pytorch3d).

    Args:
        xyz : (B, N, 3)  float tensor
        K   : number of points to sample

    Returns:
        sampled_xyz : (B, K, 3)
        indices     : (B, K)   long tensor
    """
    B, N, _ = xyz.shape
    K = min(K, N)
    device = xyz.device

    indices   = torch.zeros(B, K, dtype=torch.long,  device=device)
    distances = torch.full((B, N), float('inf'),      device=device)

    # pick a random starting point per batch element
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    for i in range(K):
        indices[:, i] = farthest
        centroid = xyz[torch.arange(B, device=device), farthest].unsqueeze(1)  # (B,1,3)
        dist = ((xyz - centroid) ** 2).sum(-1)                                  # (B,N)
        distances = torch.minimum(distances, dist)
        farthest = distances.argmax(dim=1)

    sampled_xyz = xyz[torch.arange(B, device=device).unsqueeze(1), indices]    # (B,K,3)
    return sampled_xyz, indices


class _KNNResult:
    """Mimics the pytorch3d knn_points return value (.idx attribute)."""
    def __init__(self, idx):
        self.idx = idx


def knn_points(query: torch.Tensor, ref: torch.Tensor, K: int, return_nn: bool = False):
    """
    K-nearest neighbours (pure PyTorch, no pytorch3d).

    Args:
        query : (B, M, 3)
        ref   : (B, N, 3)
        K     : number of neighbours

    Returns:
        object with .idx of shape (B, M, K)
    """
    B, M, _ = query.shape
    _, N, _ = ref.shape
    K = min(K, N)

    # (B, M, N) pairwise squared distances — chunked to save memory
    # For large N we chunk over M to keep peak memory low
    chunk = max(1, 512 // max(N // 512, 1))   # heuristic chunk size
    idx_list = []
    for b in range(B):
        q_b = query[b]          # (M, 3)
        r_b = ref[b]            # (N, 3)
        idx_b_list = []
        for start in range(0, M, chunk):
            end  = min(start + chunk, M)
            q_c  = q_b[start:end]                           # (c, 3)
            dist = ((q_c.unsqueeze(1) - r_b.unsqueeze(0)) ** 2).sum(-1)  # (c, N)
            _, ki = dist.topk(K, dim=-1, largest=False, sorted=False)     # (c, K)
            idx_b_list.append(ki)
        idx_list.append(torch.cat(idx_b_list, dim=0))       # (M, K)
    idx = torch.stack(idx_list, dim=0)                       # (B, M, K)
    return _KNNResult(idx)


# ── Load stats ────────────────────────────────────────────────────────────────
compare_stats = torch.load("pct_compare_stats.pt", weights_only=False)
CFG           = compare_stats["cfg"]
mdca_res      = compare_stats["results"]["FL+MDCA"]
ce_res        = compare_stats["results"]["CE"]
res_3a        = torch.load("cell3a_results.pt", weights_only=False)

print(f"FL+MDCA teacher | test_acc={mdca_res['test_acc']:.2f}% | "
      f"test_ECE={mdca_res['test_ece']:.2f}%")
print(f"Cell-3A         | LFT zero-shot test={res_3a['test_acc_zs']:.2f}% | "
      f"teacher test={res_3a['test_acc_pct']:.2f}%")

BLOCK      = dict(src=3, tgt=9, name="BlockB_OA3to9")
N_REPLACED = BLOCK["tgt"] - BLOCK["src"]
FW_K       = res_3a["fw_k"]

assert res_3a["block"]["src"] == BLOCK["src"] and res_3a["block"]["tgt"] == BLOCK["tgt"], \
    "Block mismatch with Cell 3A"

FT_EPOCHS  = 250
FT_LR      = 1e-4

# ── Memory-safe overrides ─────────────────────────────────────────────────────
# Reduced batch size and workers to prevent Linux OOM-killer from terminating
# the process (original was 64 / CFG["num_workers"]).
BATCH_SIZE  = 32
NUM_WORKERS = 2

# ══════════════════════════════════════════════════════════════════════════════
#  FL+MDCA LOSS  (same combo used for teacher, ensures consistency)
#  L_total = L_FL + beta * L_MDCA  (paper Eq.7)
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
    """Eq.6: L_MDCA = (1/K) * sum_j | mean_i s_i[j] - mean_i q_i[j] |"""
    def __init__(self, n_classes):
        super().__init__()
        self.K = n_classes
    def forward(self, logits, target):
        probs = F.softmax(logits, dim=-1)
        avg_conf  = probs.mean(dim=0)
        avg_count = F.one_hot(target, num_classes=self.K).float().mean(dim=0)
        return torch.abs(avg_conf - avg_count).mean()

class FocalMDCALoss(nn.Module):
    """L_total = L_FL + beta * L_MDCA  (paper Eq.7)"""
    def __init__(self, n_classes, gamma=2.0, beta=10.0):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma)
        self.mdca  = MDCALoss(n_classes)
        self.beta  = beta
    def forward(self, logits, target):
        l_fl   = self.focal(logits, target)
        l_mdca = self.mdca(logits, target)
        return l_fl + self.beta * l_mdca, l_fl.detach(), l_mdca.detach()

# ══════════════════════════════════════════════════════════════════════════════
#  CALIBRATION METRICS (ECE + SCE) — tracking finetuning effect
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def compute_ece(conf, pred, lbl, n_bins=15):
    conf=conf.cpu().numpy(); pred=pred.cpu().numpy(); lbl=lbl.cpu().numpy()
    edges=np.linspace(0,1,n_bins+1); N=len(lbl); ece=0.0
    for i in range(n_bins):
        lo,hi=edges[i],edges[i+1]
        mask=(conf>lo)&(conf<=hi) if i>0 else (conf>=lo)&(conf<=hi)
        if mask.sum()==0: continue
        ece += (mask.sum()/N)*abs((pred[mask]==lbl[mask]).mean()-conf[mask].mean())
    return 100*ece

@torch.no_grad()
def compute_sce(probs, lbl, n_bins=15):
    probs=probs.cpu().numpy(); lbl=lbl.cpu().numpy()
    N,K=probs.shape; edges=np.linspace(0,1,n_bins+1); sce=0.0
    for j in range(K):
        cj=probs[:,j]; ij=(lbl==j).astype(np.float32)
        for i in range(n_bins):
            lo,hi=edges[i],edges[i+1]
            mask=(cj>lo)&(cj<=hi) if i>0 else (cj>=lo)&(cj<=hi)
            if mask.sum()==0: continue
            sce += (mask.sum()/N)*abs(ij[mask].mean()-cj[mask].mean())
    return 1000*(sce/K)

@torch.no_grad()
def evaluate(model, dl, label="", return_calib=False):
    model.eval(); ok=tot=0
    all_conf,all_pred,all_lbl,all_probs=[],[],[],[]
    for pts,lbl in dl:
        pts=pts.to(DEVICE,non_blocking=True); lbl=lbl.to(DEVICE,non_blocking=True)
        with autocast(DEVICE):
            logits=model(pts)
        probs=F.softmax(logits.float(),dim=-1); conf,pred=probs.max(-1)
        ok+=(pred==lbl).sum().item(); tot+=lbl.size(0)
        all_conf.append(conf); all_pred.append(pred); all_lbl.append(lbl); all_probs.append(probs)
    acc=100.0*ok/tot
    if label: print(f"  [{label}]  {acc:.2f}%")
    if return_calib:
        ct=torch.cat(all_conf); pt=torch.cat(all_pred)
        lt=torch.cat(all_lbl);  pr=torch.cat(all_probs)
        return acc, tot, compute_ece(ct,pt,lt), compute_sce(pr,lt)
    return acc, tot

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
        self.n=n_pts; self.augment=augment; self.drop=in_dropout; self.seed_offset=seed_offset
        d,l=[],[]
        for f in files:
            with h5py.File(f) as h:
                d.append(h["data"][:].astype(np.float32))
                l.append(h["label"][:].flatten().astype(np.int64))
        self.data=np.concatenate(d); self.labels=np.concatenate(l)
        print(f"  Loaded {len(self.data):,} clouds  augment={augment}")
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        pts=self.data[i].copy()
        rng=np.random.RandomState(i+self.seed_offset)
        idx=rng.choice(pts.shape[0],self.n,replace=False)
        pts=pc_normalize(pts[idx])
        if self.augment:
            pts+=np.random.uniform(-0.2,0.2,(1,3)).astype(np.float32)
            pts*=np.random.uniform(0.67,1.5,(1,3)).astype(np.float32)
            n=pts.shape[0]; dd=int(n*self.drop*np.random.rand())
            if dd>0:
                di=np.random.choice(n,dd,replace=False)
                pts[di]=pts[np.random.choice(n,dd,replace=True)]
        return torch.from_numpy(pts), int(self.labels[i])

def strat_split(labels, ratio=0.20, seed=42):
    rng=np.random.RandomState(seed); tr,va=[],[]
    for c in np.unique(labels):
        idx=np.where(labels==c)[0]; rng.shuffle(idx)
        nv=max(1,int(len(idx)*ratio))
        va+=idx[:nv].tolist(); tr+=idx[nv:].tolist()
    return np.array(tr), np.array(va)

class SubsetDS(Dataset):
    def __init__(self, base, indices, augment, in_dropout=0.0):
        self.base=base; self.idx=indices; self.augment=augment; self.drop=in_dropout
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        pts,label=self.base[self.idx[i]]
        if isinstance(pts,torch.Tensor): pts=pts.numpy()
        pts=pts.copy().astype(np.float32)
        if self.augment:
            pts+=np.random.uniform(-0.2,0.2,(1,3)).astype(np.float32)
            pts*=np.random.uniform(0.67,1.5,(1,3)).astype(np.float32)
            n=pts.shape[0]; dd=int(n*self.drop*np.random.rand())
            if dd>0:
                di=np.random.choice(n,dd,replace=False)
                pts[di]=pts[np.random.choice(n,dd,replace=True)]
        return torch.from_numpy(pts), label

print("\nBuilding datasets ...")
train_pool=MN40(TRAIN_FILES,CFG["n_pts"],augment=False,seed_offset=0)
test_pool =MN40(TEST_FILES, CFG["n_pts"],augment=False,seed_offset=99999)
tr_idx,va_idx=strat_split(train_pool.labels,CFG.get("val_ratio",0.2),CFG.get("seed",42))
train_ds=SubsetDS(train_pool,tr_idx,augment=True,in_dropout=CFG.get("in_dropout",0.0))
val_ds  =SubsetDS(train_pool,va_idx,augment=False)

# pin_memory disabled when num_workers is low — reduces memory overhead
DLK = dict(
    num_workers=NUM_WORKERS,
    pin_memory=(NUM_WORKERS > 0 and DEVICE == "cuda"),
    persistent_workers=(NUM_WORKERS > 0),
)
train_dl=DataLoader(train_ds, BATCH_SIZE, shuffle=True,  drop_last=True, **DLK)
val_dl  =DataLoader(val_ds,   BATCH_SIZE, shuffle=False, **DLK)
test_dl =DataLoader(test_pool, BATCH_SIZE, shuffle=False, **DLK)
print(f"Train:{len(tr_idx):,}  Val:{len(va_idx):,}  Test:{len(test_pool):,}")
print(f"Batch size={BATCH_SIZE}  num_workers={NUM_WORKERS}")

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL BLOCKS
# ══════════════════════════════════════════════════════════════════════════════
class LBR(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(i,o,bias=False),nn.BatchNorm1d(o),nn.ReLU(True))
    def forward(self, x):
        s=x.shape; return self.net(x.reshape(-1,s[-1])).reshape(*s[:-1],-1)

class SGModule(nn.Module):
    def __init__(self, nout, k, din, dout):
        super().__init__()
        self.nout=nout; self.k=k; self.l1=LBR(din*2,dout); self.l2=LBR(dout,dout)
    def forward(self, xyz, feat):
        B,N,_=xyz.shape; no=min(self.nout,N); k=min(self.k,N)
        sxyz,fi=sample_farthest_points(xyz,K=no)
        ki=knn_points(sxyz,xyz,K=k,return_nn=False).idx; C=feat.shape[-1]
        nbr=feat.unsqueeze(1).expand(-1,no,-1,-1).gather(2,ki.unsqueeze(-1).expand(-1,-1,-1,C))
        ctr=feat.gather(1,fi.unsqueeze(-1).expand(-1,-1,C))
        agg=torch.cat([nbr-ctr.unsqueeze(2).expand_as(nbr),ctr.unsqueeze(2).expand_as(nbr)],-1)
        Bo,No,K,C2=agg.shape; out=self.l2(self.l1(agg.reshape(Bo*No*K,C2)))
        return sxyz, out.reshape(Bo,No,K,-1).max(2).values

class NE(nn.Module):
    def __init__(self, ns1, ns2, k, D):
        super().__init__()
        self.lbr=LBR(3,64); self.sg1=SGModule(ns1,k,64,64); self.sg2=SGModule(ns2,k,64,D)
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
        self.ne  =NE(cfg["n_sg1"],cfg["n_sg2"],cfg["k_sg"],D)
        self.oas =nn.ModuleList([OA(D,da) for _ in range(L)])
        self.proj=nn.Linear(D*L,1024,bias=False)
        self.head=nn.Sequential(
            nn.Linear(2048,256,bias=False),nn.BatchNorm1d(256),nn.ReLU(True),nn.Dropout(cfg["drop_rate"]),
            nn.Linear(256, 256,bias=False),nn.BatchNorm1d(256),nn.ReLU(True),nn.Dropout(cfg["drop_rate"]),
            nn.Linear(256,cfg["n_classes"]))
    def forward(self, xyz):
        _,h=self.ne(xyz); outs=[]
        for oa in self.oas: h=oa(h); outs.append(h)
        f=self.proj(torch.cat(outs,-1)); g=torch.cat([f.max(1).values,f.mean(1)],-1)
        return self.head(g)

class FWVelocityEstimator(nn.Module):
    def __init__(self, D, da):
        super().__init__()
        self.D=D
        self.t_mlp=nn.Sequential(nn.Linear(1,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,2*D))
        self.norm=nn.LayerNorm(D); self.oa=OA(D,da)
    def forward(self, xt, t):
        out=self.t_mlp(t.unsqueeze(-1).float()); scale,shift=out.chunk(2,dim=-1)
        h=self.norm(xt)*(1.0+scale.unsqueeze(1))+shift.unsqueeze(1)
        h=self.oa(h); return h-xt
    def step(self, xt, t_cur, t_nxt):
        d=(t_nxt-t_cur).view(-1,1,1); t_m=(t_cur+t_nxt)/2
        v1=self.forward(xt,t_cur); v2=self.forward(xt+0.5*d*v1,t_m)
        return xt+d*v2
    def transport(self, x0, k=3):
        x=x0; B=x0.shape[0]
        for i in range(k):
            t_c=torch.full((B,),i/k,    device=x.device,dtype=x.dtype)
            t_n=torch.full((B,),(i+1)/k,device=x.device,dtype=x.dtype)
            x=self.step(x,t_c,t_n)
        return x

class LFT_PCT(nn.Module):
    def __init__(self, cfg, fw_net, src, tgt, fw_k=3):
        super().__init__()
        D=cfg["embed_dim"]; da=cfg["da"]; L=cfg["n_attn"]
        self.src,self.tgt,self.fw_k,self.L=src,tgt,fw_k,L
        self.kept=[i for i in range(L) if not (src<=i<tgt)]
        self.n_slots=len(self.kept)+1
        self.ne  =NE(cfg["n_sg1"],cfg["n_sg2"],cfg["k_sg"],D)
        self.oas =nn.ModuleList([OA(D,da) for _ in self.kept])
        self.fw  =fw_net
        self.proj=nn.Linear(D*self.n_slots,1024,bias=False)
        self.head=nn.Sequential(
            nn.Linear(2048,256,bias=False),nn.BatchNorm1d(256),nn.ReLU(True),nn.Dropout(cfg["drop_rate"]),
            nn.Linear(256, 256,bias=False),nn.BatchNorm1d(256),nn.ReLU(True),nn.Dropout(cfg["drop_rate"]),
            nn.Linear(256,cfg["n_classes"]))
    def forward(self, xyz):
        _,h=self.ne(xyz); outs=[]; oa_i=0; B=h.shape[0]
        for i in range(self.L):
            if i==self.src:
                if not torch.is_grad_enabled():
                    h=self.fw.transport(h,k=self.fw_k)
                else:
                    for s in range(self.fw_k):
                        t_c=torch.full((B,),s/self.fw_k,    device=h.device)
                        t_n=torch.full((B,),(s+1)/self.fw_k,device=h.device)
                        h=self.fw.step(h,t_c,t_n)
                outs.append(h)
            elif self.src<i<self.tgt:
                pass
            else:
                h=self.oas[oa_i](h); outs.append(h); oa_i+=1
        f=self.proj(torch.cat(outs,-1)); g=torch.cat([f.max(1).values,f.mean(1)],-1)
        return self.head(g)

def nparams(m): return sum(p.numel() for p in m.parameters())

def transplant_weights(lft, teacher_sd, kept, src, tgt, D, L):
    l_sd=lft.state_dict()
    for k in list(l_sd.keys()):
        if k.startswith("ne."):   l_sd[k]=teacher_sd[k]
        if k.startswith("head."): l_sd[k]=teacher_sd[k]
    for new_i,old_i in enumerate(kept):
        pn,po=f"oas.{new_i}.",f"oas.{old_i}."
        for k in list(l_sd.keys()):
            if k.startswith(pn): l_sd[k]=teacher_sd[po+k[len(pn):]]
    lft.load_state_dict(l_sd)
    t_proj=teacher_sd["proj.weight"]
    blocks=[t_proj[:,i*D:(i+1)*D] for i in range(L)]
    flow_block=sum(blocks[i] for i in range(src,tgt))
    new_proj_w=torch.cat([blocks[i] for i in range(0,src)]+[flow_block]+[blocks[i] for i in range(tgt,L)],dim=1)
    lft.proj.weight.data.copy_(new_proj_w)
    print(f"  Transplanted: NE, {len(kept)} kept OA layers, head, proj ({len(kept)+1} slots)")

# ══════════════════════════════════════════════════════════════════════════════
#  BUILD MODEL
# ══════════════════════════════════════════════════════════════════════════════
D=CFG["embed_dim"]; da=CFG["da"]; L=CFG["n_attn"]

teacher=PCT(CFG).to(DEVICE)
teacher.load_state_dict(torch.load("pct_mdca.pth",map_location=DEVICE,weights_only=True))
teacher.eval()
for p in teacher.parameters(): p.requires_grad_(False)
n_pct=nparams(teacher)

fw   =FWVelocityEstimator(D,da).to(DEVICE)
model=LFT_PCT(CFG,fw,BLOCK["src"],BLOCK["tgt"],fw_k=FW_K).to(DEVICE)
transplant_weights(model,teacher.state_dict(),model.kept,BLOCK["src"],BLOCK["tgt"],D,L)
model.fw.load_state_dict(torch.load("lft_blockB_vel.pth",map_location=DEVICE,weights_only=True))

n_total=nparams(model); n_fw_params=nparams(model.fw)
print(f"\n  LFT-PCT total: {n_total:,}  (PCT: {n_pct:,},  saving {100*(n_pct-n_total)/n_pct:.1f}%)")

pct_flops=flops_pct_teacher(CFG); lft_flops=flops_lft_pct(CFG,BLOCK,fw_k=FW_K)
LFT_FLOPS_PER_SAMPLE      =lft_flops["total"]
LFT_TRAIN_FLOPS_PER_SAMPLE=3*LFT_FLOPS_PER_SAMPLE
flop_save_pct=100*(pct_flops["total"]-lft_flops["total"])/pct_flops["total"]
print(f"  FLOPs/sample: PCT={human(pct_flops['total'])} | LFT={human(lft_flops['total'])} | saving={flop_save_pct:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — END-TO-END FINETUNING with FL+MDCA loss
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print(f"  PHASE 3 — End-to-End Finetuning  ({FT_EPOCHS} epochs)")
print(f"  Loss    : FocalLoss(gamma={CFG['focal_gamma']}) + {CFG['mdca_beta']}*MDCA")
print(f"  Teacher : FL+MDCA calibrated PCT  (pct_mdca.pth)")
print(f"  Block   : OA[{BLOCK['src']}..{BLOCK['tgt']-1}]  ({N_REPLACED} layers -> 1 FW net)")
print("="*65)

for p in model.parameters(): p.requires_grad_(True)
print(f"\n  Trainable params: {nparams(model):,}")

opt_ft   =torch.optim.AdamW(model.parameters(),lr=FT_LR,weight_decay=1e-4)
sched_ft =torch.optim.lr_scheduler.CosineAnnealingLR(opt_ft,FT_EPOCHS,eta_min=1e-6)
# FL+MDCA — consistent with teacher training
crit     =FocalMDCALoss(n_classes=CFG["n_classes"],gamma=CFG["focal_gamma"],beta=CFG["mdca_beta"])
scaler_p3=GradScaler(DEVICE)

ft_tr_losses,ft_fl_losses,ft_mdca_losses=[],[],[]
ft_tr_accs,ft_val_accs,ft_val_eces,ft_ep_times=[],[],[],[]
ft_train_flops_cumulative,ft_val_flops_cumulative=[],[]
total_ft_train_flops=0.0; total_ft_val_flops=0.0
best_val_ft=0.0
t_ft_start=time.perf_counter()

print(f"\n  {'Ep':>4} | {'Loss':>7} | {'FL':>7} | {'MDCA':>7} | "
      f"{'TrAcc':>7} | {'ValAcc':>8} | {'ECE':>6} | {'Best':>7} | {'Time':>7}")
print(f"  {'-'*90}")

for ep in range(1,FT_EPOCHS+1):
    model.train(); t0=time.perf_counter()
    ep_loss=ep_fl=ep_mdca=ep_ok=ep_n=0
    for pts,lbl in train_dl:
        pts=pts.to(DEVICE,non_blocking=True); lbl=lbl.to(DEVICE,non_blocking=True)
        opt_ft.zero_grad(set_to_none=True)
        with autocast(DEVICE):
            logits=model(pts)
            loss,l_fl,l_mdca=crit(logits,lbl)
        scaler_p3.scale(loss).backward()
        scaler_p3.unscale_(opt_ft)
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        scaler_p3.step(opt_ft); scaler_p3.update()
        with torch.no_grad():
            ep_ok+=(logits.argmax(1)==lbl).sum().item()
        ep_loss+=loss.item()*lbl.size(0); ep_fl+=l_fl.item()*lbl.size(0)
        ep_mdca+=l_mdca.item()*lbl.size(0); ep_n+=lbl.size(0)
        total_ft_train_flops+=LFT_TRAIN_FLOPS_PER_SAMPLE*lbl.size(0)

    sched_ft.step()
    ep_time=time.perf_counter()-t0; cum_time=time.perf_counter()-t_ft_start
    tr_loss=ep_loss/ep_n; tr_acc=100.0*ep_ok/ep_n
    tr_fl=ep_fl/ep_n; tr_mdca=ep_mdca/ep_n
    val_acc,n_val,val_ece,val_sce=evaluate(model,val_dl,return_calib=True)
    total_ft_val_flops+=LFT_FLOPS_PER_SAMPLE*n_val

    if val_acc>best_val_ft:
        best_val_ft=val_acc
        torch.save(model.state_dict(),"lft_ft_blockB.pth")

    ft_tr_losses.append(tr_loss); ft_fl_losses.append(tr_fl); ft_mdca_losses.append(tr_mdca)
    ft_tr_accs.append(tr_acc); ft_val_accs.append(val_acc); ft_val_eces.append(val_ece)
    ft_ep_times.append(ep_time)
    ft_train_flops_cumulative.append(total_ft_train_flops)
    ft_val_flops_cumulative.append(total_ft_val_flops)

    if ep%10==0 or ep==1:
        print(f"  {ep:>4} | {tr_loss:>7.4f} | {tr_fl:>7.4f} | {tr_mdca:>7.5f} | "
              f"{tr_acc:>6.2f}% | {val_acc:>7.2f}% | {val_ece:>5.2f}% | "
              f"{best_val_ft:>6.2f}% | {ep_time:>6.1f}s | {cum_time/60:>5.1f}m")

total_ft_time=time.perf_counter()-t_ft_start
model.load_state_dict(torch.load("lft_ft_blockB.pth",map_location=DEVICE,weights_only=True))

# ══════════════════════════════════════════════════════════════════════════════
#  FINAL TEST EVALUATION — accuracy + calibration for all models
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  FINAL TEST EVALUATION  (after Phase 3 finetuning)")
print("="*65)

test_acc_ft, n_test, test_ece_ft, test_sce_ft = evaluate(model,   test_dl, return_calib=True)
test_acc_pct,_,      test_ece_pct,test_sce_pct= evaluate(teacher, test_dl, return_calib=True)
print(f"  [LFT finetuned Test]  acc={test_acc_ft:.2f}%  ECE={test_ece_ft:.2f}%  SCE={test_sce_ft:.4f}e-3")
print(f"  [FL+MDCA teacher Test] acc={test_acc_pct:.2f}%  ECE={test_ece_pct:.2f}%  SCE={test_sce_pct:.4f}e-3")

test_acc_zs   = res_3a["test_acc_zs"]
test_flops_pct    = pct_flops["total"]*n_test
test_flops_lft_zs = lft_flops["total"]*n_test
test_flops_lft_ft = lft_flops["total"]*n_test
flop_save_abs     = test_flops_pct-test_flops_lft_ft
combined_lft_train= res_3a["total_p1_train_flops"]+total_ft_train_flops

# ══════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
fig.suptitle(
    f"Phase 3 — Finetuning  OA[{BLOCK['src']}..{BLOCK['tgt']-1}]  (FL+MDCA loss)\n"
    f"Test: LFT-FT={test_acc_ft:.2f}% ECE={test_ece_ft:.2f}% | "
    f"ZS={test_acc_zs:.2f}% | Teacher={test_acc_pct:.2f}% ECE={test_ece_pct:.2f}%\n"
    f"FLOP saving vs PCT: {100*flop_save_abs/test_flops_pct:.1f}%",
    fontsize=9, fontweight="bold")

axes[0,0].plot(ft_fl_losses,   "steelblue", lw=2, label="Focal")
axes[0,0].plot(ft_mdca_losses, "purple",    lw=2, label="MDCA (raw)")
axes[0,0].set(title="Loss Components", xlabel="Epoch", ylabel="Loss")
axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

axes[0,1].plot(ft_tr_accs,  "steelblue", lw=2, label="Train")
axes[0,1].plot(ft_val_accs, "coral",     lw=2, label="Val")
axes[0,1].axhline(best_val_ft,  color="coral", ls=":",  lw=1.5, label=f"best val={best_val_ft:.1f}%")
axes[0,1].axhline(test_acc_pct, color="gray",  ls="--", lw=1.5, label=f"teacher={test_acc_pct:.1f}%")
axes[0,1].axhline(test_acc_zs,  color="orange",ls=":",  lw=1.5, label=f"zero-shot={test_acc_zs:.1f}%")
axes[0,1].set(title="Accuracy", xlabel="Epoch", ylabel="%")
axes[0,1].legend(fontsize=7); axes[0,1].grid(alpha=0.3)

axes[0,2].plot(ft_val_eces, "darkgreen", lw=2)
axes[0,2].axhline(mdca_res["test_ece"], color="gray", ls="--", lw=1.5,
                   label=f"teacher ECE={mdca_res['test_ece']:.2f}%")
axes[0,2].axhline(ce_res["test_ece"],   color="red",  ls=":",  lw=1.5,
                   label=f"CE baseline ECE={ce_res['test_ece']:.2f}%")
axes[0,2].set(title="Val ECE (FL+MDCA finetuned)", xlabel="Epoch", ylabel="%")
axes[0,2].legend(fontsize=7); axes[0,2].grid(alpha=0.3)

axes[1,0].plot([f/1e9 for f in ft_train_flops_cumulative], "teal", lw=2, label="LFT-FT train GFLOPs")
axes[1,0].set(title="Cumulative Training FLOPs", xlabel="Epoch", ylabel="GFLOPs")
axes[1,0].legend(); axes[1,0].grid(alpha=0.3)

# Reliability diagram — LFT finetuned vs FL+MDCA teacher
n_bins=15; edges=np.linspace(0,1,n_bins+1); centers=(edges[:-1]+edges[1:])/2
for ax_i, (m, lbl, col) in enumerate([
        (model,   f"LFT-FT (ECE={test_ece_ft:.2f}%)",    "coral"),
        (teacher, f"FL+MDCA teacher (ECE={test_ece_pct:.2f}%)", "steelblue")]):
    all_conf,all_pred,all_lbl=[],[],[]
    with torch.no_grad():
        for pts,lb in test_dl:
            pts=pts.to(DEVICE); lb=lb.to(DEVICE)
            with autocast(DEVICE):
                logits=m(pts)
            probs=F.softmax(logits.float(),dim=-1); conf,pred=probs.max(-1)
            all_conf.append(conf.cpu()); all_pred.append(pred.cpu()); all_lbl.append(lb.cpu())
    conf_t=torch.cat(all_conf).numpy(); pred_t=torch.cat(all_pred).numpy(); lbl_t=torch.cat(all_lbl).numpy()
    accs=[]
    for i in range(n_bins):
        lo,hi=edges[i],edges[i+1]
        mask=(conf_t>lo)&(conf_t<=hi) if i>0 else (conf_t>=lo)&(conf_t<=hi)
        accs.append((pred_t[mask]==lbl_t[mask]).mean() if mask.sum()>0 else 0)
    axes[1,ax_i+1].bar(centers,accs,width=1/n_bins,color=col,alpha=0.7,label="Accuracy")
    axes[1,ax_i+1].plot([0,1],[0,1],"k--",label="Perfect calib.")
    axes[1,ax_i+1].set(title=f"Reliability — {lbl}",xlabel="Confidence",ylabel="Accuracy")
    axes[1,ax_i+1].legend(fontsize=7)

plt.tight_layout()
plt.savefig("cell3b_phase3_finetune.png",dpi=150,bbox_inches="tight")
plt.show(); print("Saved -> cell3b_phase3_finetune.png")

# ══════════════════════════════════════════════════════════════════════════════
#  FINAL SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
saving=n_pct-n_total
print(f"\n{'='*65}")
print(f"  CELL 3B — FINAL SUMMARY  (FL+MDCA throughout)")
print(f"{'='*65}")
print(f"  {'Metric':<44} {'Value':>16}")
print(f"  {'-'*62}")
print(f"  {'Block replaced':<44} {'OA[3..8] (6 layers)':>16}")
print(f"  {'Loss (finetuning)':<44} {'FL+MDCA':>16}")
print(f"  {'Teacher source':<44} {'pct_mdca.pth':>16}")
print(f"  {'-'*62}")
print(f"  {'Test acc — CE baseline (no flow)':<44} {ce_res['test_acc']:>15.2f}%")
print(f"  {'Test acc — FL+MDCA teacher (no flow)':<44} {test_acc_pct:>15.2f}%")
print(f"  {'Test acc — LFT zero-shot (3A)':<44} {test_acc_zs:>15.2f}%")
print(f"  {'Test acc — LFT finetuned (3B)':<44} {test_acc_ft:>15.2f}%")
print(f"  {'  Recovery over zero-shot':<44} {test_acc_ft-test_acc_zs:>+15.2f}%")
print(f"  {'  Gap vs FL+MDCA teacher':<44} {test_acc_ft-test_acc_pct:>+15.2f}%")
print(f"  {'-'*62}")
print(f"  {'ECE — CE baseline':<44} {ce_res['test_ece']:>15.2f}%")
print(f"  {'ECE — FL+MDCA teacher':<44} {test_ece_pct:>15.2f}%")
print(f"  {'ECE — LFT finetuned':<44} {test_ece_ft:>15.2f}%")
print(f"  {'SCE — CE baseline (1e-3)':<44} {ce_res['test_sce']:>15.4f}")
print(f"  {'SCE — FL+MDCA teacher (1e-3)':<44} {test_sce_pct:>15.4f}")
print(f"  {'SCE — LFT finetuned (1e-3)':<44} {test_sce_ft:>15.4f}")
print(f"  {'-'*62}")
print(f"  {'FLOPs/sample — PCT teacher':<44} {human(pct_flops['total']):>16}")
print(f"  {'FLOPs/sample — LFT-PCT':<44} {human(lft_flops['total']):>16}")
print(f"  {'  FLOP reduction (fwd pass)':<44} {flop_save_pct:>15.1f}%")
print(f"  {'Params — PCT teacher':<44} {n_pct:>16,}")
print(f"  {'Params — LFT-PCT':<44} {n_total:>16,}  ({100*saving/n_pct:.1f}% smaller)")
print(f"  {'-'*62}")
print(f"  {'Train FLOPs — Phase 1 (3A)':<44} {human(res_3a['total_p1_train_flops']):>16}")
print(f"  {'Train FLOPs — Phase 3 (3B)':<44} {human(total_ft_train_flops):>16}")
print(f"  {'Train FLOPs — LFT combined (P1+P3)':<44} {human(combined_lft_train):>16}")
print(f"{'='*65}")

# Save
torch.save({
    "cfg": CFG, "block": BLOCK, "fw_k": FW_K, "ft_epochs": FT_EPOCHS,
    "n_lft_total": n_total, "n_fw": n_fw_params, "n_pct": n_pct,
    "ft_tr_losses": ft_tr_losses, "ft_fl_losses": ft_fl_losses,
    "ft_mdca_losses": ft_mdca_losses,
    "ft_tr_accs": ft_tr_accs, "ft_val_accs": ft_val_accs,
    "ft_val_eces": ft_val_eces, "ft_ep_times": ft_ep_times,
    "total_ft_time_s": total_ft_time, "best_val_ft": best_val_ft,
    "test_acc_ft": test_acc_ft, "test_ece_ft": test_ece_ft, "test_sce_ft": test_sce_ft,
    "test_acc_zs": test_acc_zs,
    "test_acc_pct": test_acc_pct, "test_ece_pct": test_ece_pct, "test_sce_pct": test_sce_pct,
    "ce_test_acc": ce_res["test_acc"], "ce_test_ece": ce_res["test_ece"],
    "teacher_source": "FL+MDCA",
    "pct_flops": pct_flops, "lft_flops": lft_flops,
    "n_test": n_test,
    "test_flops_pct": test_flops_pct, "test_flops_lft_ft": test_flops_lft_ft,
    "ft_train_flops_cumulative": ft_train_flops_cumulative,
    "total_ft_train_flops": total_ft_train_flops, "total_ft_val_flops": total_ft_val_flops,
    "total_p1_train_flops": res_3a["total_p1_train_flops"],
    "combined_lft_train_flops": combined_lft_train,
}, "cell3b_results.pt")
print("\nSaved: lft_ft_blockB.pth | cell3b_results.pt | cell3b_phase3_finetune.png")
print("Cell 3B (finetuning) complete.")