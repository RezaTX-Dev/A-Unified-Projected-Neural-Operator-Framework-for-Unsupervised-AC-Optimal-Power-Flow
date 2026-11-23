"""
pno_acopf_tables.py  --  ONE self-contained file
  Result tables of "A Unified Projected Neural Operator Framework for Unsupervised AC Optimal Power Flow:
  A Comparative Study of Spectral, Branch-Trunk, and Graph Architectures".
  Loads standard PYPOWER/MATPOWER test networks, creates randomized load scenarios, and solves a NEW AC-OPF for every retained scenario using PYPOWER. The resulting OPF solutions are used as training labels.

    pip install torch numpy scipy pandas pypower
"""
import argparse, copy, csv, math, os, pickle, sys, time, zlib, warnings, platform, datetime
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import scipy
from scipy import stats as sst
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

warnings.filterwarnings("ignore")

# =====================================================================================
#  >>>>>>>>>>>>>>>>>>>>>  CHANGE ONLY THESE VARIABLES  <<<<<<<<<<<<<<<<<<<<<
# =====================================================================================
CASE_SELECT = ["case14"]
N_DATA      = 3000
EPOCHS      = 100
N_SEEDS     = 3
# =====================================================================================

# ---- standard PYPOWER/MATPOWER test cases; no PyTorch Geometric / generated AC-OPF scenarios ----
CASES = {"case14": "case14", "case30": "case30", "case57": "case57", "case118": "case118", "case300": "case300"}
LOAD_MIN, LOAD_MAX = 0.70, 1.30
BUS_PERTURB_STD, Q_PERTURB_STD = 0.12, 0.10
REGIONAL_CORRELATION = 0.35
OPF_MAX_RETRIES = 12
OPF_VERBOSE = 0
OPF_OPTIONS = {"VERBOSE": OPF_VERBOSE, "OUT_ALL": 0, "OPF_ALG": 560}
BASE_MVA = 100.0
COST_IN_MW = False
LINE  = dict(bfr=2, bto=3, r=4, x=5, rate=6)
TRAFO = dict(r=2, x=3, rate=4, tap=7, shift=8, bfr=9, bto=10)
GEN   = dict(pmin=2, pmax=3, qmin=5, qmax=6, c2=8, c1=9, c0=10)
SHUNT = dict(bs=0, gs=1)

# ---- fixed settings ----
RUN_MODE, FORCE_REGEN, QUICK_TEST = "all", False, False
SYSTEMS   = list(CASE_SELECT)
N_SAMPLES = {s: N_DATA for s in SYSTEMS}
TEST_MAX, VAL_MAX, TEST_FRAC, VAL_FRAC = 1500, 1000, 0.15, 0.15
ABL_SEEDS, ABL_SYSTEM = 1, None
EXTRAP_SEEDS = 1
EXTRAP_Q = 0.80
PATIENCE, MIN_STEPS = 25, 20
MIN_REL_GAIN = 1e-3
EARLY_STOP_AFTER = 0.20
BATCH, LR, LR_MIN, WD = 256, 1e-3, 1e-5, 1e-5
CLIP = 1.0
W_LAT, L_BLK, M_MODES = 64, 4, 12
WNO_LEVELS = 2
DON_P, DON_TRUNK_W, DON_BRANCH_W = 64, 128, 256
GNO_L, GNO_KW = 3, 32
MLP_W, MLP_L = 512, 5
PROJ_IT, PROJ_EPS = 3, 1e-6
THETA_SCALE = 0.3
SWEEP_K = (0, 1, 2, 3, 4, 6)
TAU_EQ, TAU_INEQ = 1e-3, 1e-4
THETA_BASE = 1.0
LEVEL_NAMES = ["Light", "Nominal", "Heavy"]
LAM_PNO  = dict(cost=1.0, bal=10.0, flow=1.0, ineq=0.0, sup=0.0)
LAM_SOFT = dict(cost=1.0, bal=5.0,  flow=1.0, ineq=1.0, sup=0.0)
LAM_SUP  = dict(cost=0.0, bal=0.0,  flow=0.0, ineq=0.0, sup=1.0)
NEWTON_IT, PVPQ_OUTER = 8, 4
N_CH = 13
NET = ["PNO-FNO", "PNO-DeepONet", "PNO-GNO", "PNO-WNO"]
ORDER = NET + ["Unsup-FNO-Soft", "Sup-MLP"]
DISPLAY = {"AC-OPF": "PYPOWER AC-OPF reference", "Sup-MLP": "Supervised MLP", "Unsup-FNO-Soft": "Unsup. FNO (Soft)",
           "PNO-FNO": "PNO-FNO (Ours)", "PNO-DeepONet": "PNO-DeepONet", "PNO-GNO": "PNO-GNO", "PNO-WNO": "PNO-WNO"}
LABELS = {"AC-OPF": "--", "Sup-MLP": "Yes", "Unsup-FNO-Soft": "No", "PNO-FNO": "No", "PNO-DeepONet": "No", "PNO-GNO": "No", "PNO-WNO": "No"}
DATA_DIR, RES_DIR = "data_pno", "results_pno"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOLVE_CPU = False

def _probe_gpu():
    global DEVICE, SOLVE_CPU
    if DEVICE.type != "cuda": return
    try:
        x = torch.randn(8, 14, 16, device=DEVICE, requires_grad=True)
        f = torch.fft.irfft(torch.fft.rfft(x, dim=1) * (1 + 1j), n=14, dim=1)
        v = torch.polar(torch.ones(8, 14, device=DEVICE), x[..., 0]); (f.sum() + (v @ v.conj().T).real.sum() + torch.logit(x.sigmoid()).sum()).backward()
        torch.cuda.synchronize()
    except Exception as e:
        print(f"WARNING: GPU unusable with this torch build ({type(e).__name__}) -> running on CPU.")
        DEVICE = torch.device("cpu"); return
    try:
        B = min(int(BATCH), 256)
        M = torch.randn(B, 28, 28, device=DEVICE); M = (M @ M.transpose(1, 2) + torch.eye(28, device=DEVICE)).requires_grad_()
        torch.linalg.solve(M, torch.randn(B, 28, 1, device=DEVICE)).sum().backward(); torch.cuda.synchronize()
    except Exception as e:
        SOLVE_CPU = True; print(f"NOTE: torch.linalg.solve fails on this GPU ({type(e).__name__}) -> projection solves run on the CPU.")

_probe_gpu()

_SOLVE_VERIFIED = False
def batched_solve(M, b):
    global SOLVE_CPU, _SOLVE_VERIFIED
    if not M.is_cuda:
        return torch.linalg.solve(M, b)
    if SOLVE_CPU:
        return torch.linalg.solve(M.cpu(), b.cpu()).to(M.device)
    try:
        out = torch.linalg.solve(M, b)
        if not _SOLVE_VERIFIED:
            torch.cuda.synchronize(); _SOLVE_VERIFIED = True
        return out
    except Exception as e:
        SOLVE_CPU = True
        try: torch.cuda.synchronize()
        except Exception: pass
        print(f"NOTE: torch.linalg.solve failed on this GPU at runtime ({type(e).__name__}) -> projection solves now run on the CPU.")
        return torch.linalg.solve(M.cpu(), b.cpu()).to(M.device)

if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False

MODEL_CFG = {
    "PNO-FNO":        dict(arch="fno",  bounded=True,  proj_it=PROJ_IT, proj_train=True, labels=False, lam=dict(LAM_PNO)),
    "PNO-DeepONet":   dict(arch="don",  bounded=True,  proj_it=PROJ_IT, proj_train=True, labels=False, lam=dict(LAM_PNO)),
    "PNO-GNO":        dict(arch="gno",  bounded=True,  proj_it=PROJ_IT, proj_train=True, labels=False, lam=dict(LAM_PNO)),
    "PNO-WNO":        dict(arch="wno",  bounded=True,  proj_it=PROJ_IT, proj_train=True, labels=False, lam=dict(LAM_PNO)),
    "Unsup-FNO-Soft": dict(arch="fno",  bounded=False, proj_it=0,       proj_train=True, labels=False, lam=dict(LAM_SOFT)),
    "Sup-MLP":        dict(arch="mlp",  bounded=False, proj_it=0,       proj_train=True, labels=True,  lam=dict(LAM_SUP)),
}
ABLATIONS = [
    ("w/o projection layer",         dict(proj_it=0)),
    ("w/o sigmoid bounds",           dict(bounded=False)),
    ("projection at inference only", dict(proj_train=False)),
    ("single projection step (K=1)", dict(proj_it=1)),
]

def stable_seed(*p): return zlib.crc32("|".join(str(x) for x in p).encode()) % (2 ** 31)
def target_n(system): return int(N_SAMPLES.get(system, N_DATA))

# =====================================================================================
#  GRID STRUCTURE
# =====================================================================================
def load_pypower_case(system):
    """Load a standard PYPOWER test case and convert to internal bus indexing."""
    try:
        from pypower.api import case14, case30, case57, case118, case300, ext2int
    except ImportError as e:
        raise SystemExit("Install PYPOWER first: pip install pypower") from e
    funcs = {"case14": case14, "case30": case30, "case57": case57, "case118": case118, "case300": case300}
    return ext2int(funcs[system]())


def build_struct(ppc):
    """Build network tensors/arrays directly from a MATPOWER case (internal indexing)."""
    from pypower.makeYbus import makeYbus
    from pypower.idx_bus import BUS_I, BUS_TYPE, PD, QD, GS, BS, VM, VA, VMAX, VMIN
    from pypower.idx_gen import GEN_BUS, PG, QG, QMAX, QMIN, PMAX, PMIN
    from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, TAP, SHIFT, BR_STATUS
    from pypower.idx_cost import MODEL, NCOST, COST
    n = ppc["bus"].shape[0]; bus, gen, branch = ppc["bus"], ppc["gen"], ppc["branch"]
    nbase = float(ppc["baseMVA"])
    Ybus, Yf, Yt = makeYbus(nbase, bus, branch)
    Ybus, Yf, Yt = Ybus.toarray(), Yf.toarray(), Yt.toarray()
    fr, to = branch[:, F_BUS].astype(int), branch[:, T_BUS].astype(int)
    rate = branch[:, RATE_A] / nbase
    rate[~np.isfinite(rate) | (rate <= 0)] = 1e6
    active = branch[:, BR_STATUS] > 0
    rate[~active] = 1e6
    gen_bus = gen[:, GEN_BUS].astype(int)
    pmin_u, pmax_u = np.minimum(gen[:, PMIN], gen[:, PMAX])/nbase, np.maximum(gen[:, PMIN], gen[:, PMAX])/nbase
    qmin_u, qmax_u = np.minimum(gen[:, QMIN], gen[:, QMAX])/nbase, np.maximum(gen[:, QMIN], gen[:, QMAX])/nbase
    agg = lambda v: np.bincount(gen_bus, weights=v, minlength=n)
    Pmin, Pmax, Qmin, Qmax = agg(pmin_u), agg(pmax_u), agg(qmin_u), agg(qmax_u)
    chi_g = (np.bincount(gen_bus, minlength=n) > 0).astype(np.float32); gbus = np.where(chi_g > 0)[0]
    # Aggregate generator costs per bus using Pmax-share allocation; exact OPF objective is retained in scenario metadata.
    gc = ppc["gencost"]
    c2u, c1u, c0u = [], [], []
    for row in gc:
        if int(row[MODEL]) != 2: raise ValueError("Only polynomial MATPOWER generator costs (model=2) are supported")
        nc = int(row[NCOST]); coeff = row[COST:COST+nc]
        if nc >= 3: c2u.append(coeff[-3]); c1u.append(coeff[-2]); c0u.append(coeff[-1])
        elif nc == 2: c2u.append(0.); c1u.append(coeff[-2]); c0u.append(coeff[-1])
        else: c2u.append(0.); c1u.append(0.); c0u.append(coeff[-1])
    c2u, c1u, c0u = np.asarray(c2u), np.asarray(c1u), np.asarray(c0u)
    share = np.maximum(gen[:, PMAX], 1e-6); share /= np.maximum(agg(np.maximum(gen[:, PMAX], 1e-6))[gen_bus], 1e-6)
    c2, c1, c0 = (agg(v)[gbus] for v in (c2u*share**2, c1u*share, c0u))
    # MATPOWER costs are defined in MW; output generator Pg is p.u., so cost evaluation multiplies by baseMVA.
    W = np.abs(Ybus); np.fill_diagonal(W, 0); Ws = np.maximum(W, W.T)
    Wadj = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
    _, evec = np.linalg.eigh(np.diag(Ws.sum(1))-Ws)
    nz = lambda f: (f-f.min())/(f.max()-f.min()+1e-12)
    xi, xi2 = nz(evec[:,1]), nz(evec[:,2]) if n>2 else np.zeros(n)
    gdist = shortest_path(csr_matrix((Ws>0).astype(float)), unweighted=True, directed=False)
    static = np.stack([bus[:,VMIN], bus[:,VMAX], Pmin, Pmax, Qmin, Qmax, chi_g, bus[:,BUS_TYPE]/3], 1).astype(np.float32)
    slack_candidates=np.where(bus[:,BUS_TYPE]==3)[0]
    if len(slack_candidates)==0: raise RuntimeError("No REF/slack bus found in case")
    b=dict(n=n, sn=nbase, Ybus=Ybus, Yf=Yf, Yt=Yt, rate=rate, fb=fr, tb=to, gen_bus=gen_bus, gbus=gbus,
        chi_g=chi_g, Pmin=Pmin, Pmax=Pmax, Qmin=Qmin, Qmax=Qmax, Vmin=bus[:,VMIN], Vmax=bus[:,VMAX],
        c2=c2, c1=c1, c0=c0, Wadj=Wadj.astype(np.float32), xi=xi, xi2=xi2, gdist=gdist, static=static,
        slack=int(slack_candidates[0]), baseMVA=nbase, base_bus=bus.copy(), base_gen=gen.copy(), base_branch=branch.copy(), base_gencost=gc.copy())
    aux=dict(load_bus=np.arange(n), pd0=bus[:,PD]/nbase, qd0=bus[:,QD]/nbase, ppc_base=ppc)
    return b, aux


def sample_AY_from_solution(ppc_sol, b, aux):
    from pypower.idx_bus import PD, QD, VM, VA
    from pypower.idx_gen import GEN_BUS, PG, QG
    n=b["n"]; A=np.zeros((n,N_CH),np.float32); Y=np.zeros((n,4),np.float32)
    A[:,0]=ppc_sol["bus"][:,PD]/b["baseMVA"]; A[:,1]=ppc_sol["bus"][:,QD]/b["baseMVA"]
    A[:,5:]=b["static"]
    np.add.at(Y[:,0], ppc_sol["gen"][:,GEN_BUS].astype(int), ppc_sol["gen"][:,PG]/b["baseMVA"])
    np.add.at(Y[:,1], ppc_sol["gen"][:,GEN_BUS].astype(int), ppc_sol["gen"][:,QG]/b["baseMVA"])
    Y[:,2]=ppc_sol["bus"][:,VM]; Y[:,3]=np.deg2rad(ppc_sol["bus"][:,VA]-ppc_sol["bus"][b["slack"],VA])
    return A,Y


# =====================================================================================
#  METRICS / FEASIBILITY
# =====================================================================================
def _relu(x): return np.maximum(x, 0.0)

def flows(y, b):
    Vc = y[..., 2] * np.exp(1j * y[..., 3])
    If, It = Vc @ b["Yf"].T, Vc @ b["Yt"].T
    return Vc, Vc[..., b["fb"]] * np.conj(If), Vc[..., b["tb"]] * np.conj(It)

def sample_metrics(y, A, Y, b, chunk=256):
    N, gb, sn = len(y), b["gbus"], b["sn"]; fin = b["rate"] < 1e5
    keys = ["se_pg", "se_qg", "se_v", "se_th", "gap", "r_p", "r_q", "r_bal", "r_gen", "r_volt", "r_br"]
    M = {k: np.zeros(N) for k in keys}
    J = lambda pg: (b["c2"] * (pg * sn) ** 2 + b["c1"] * pg * sn + b["c0"]).sum(1)
    for s in range(0, N, chunk):
        sl = slice(s, s + chunk)
        yy, AA, YY = y[sl].astype(np.float64), A[sl].astype(np.float64), Y[sl].astype(np.float64)
        Vc, Sf, St = flows(yy, b)
        Sb = Vc * np.conj(Vc @ b["Ybus"].T)
        dP = yy[..., 0] - AA[..., 0] - Sb.real; dQ = yy[..., 1] - AA[..., 1] - Sb.imag
        M["r_p"][sl] = np.abs(dP).max(1); M["r_q"][sl] = np.abs(dQ).max(1)
        M["r_bal"][sl] = np.maximum(M["r_p"][sl], M["r_q"][sl])
        pg, qg, v = yy[:, gb, 0], yy[:, gb, 1], yy[..., 2]
        M["r_gen"][sl] = np.maximum.reduce([_relu(pg - b["Pmax"][gb]).max(1), _relu(b["Pmin"][gb] - pg).max(1),
                                            _relu(qg - b["Qmax"][gb]).max(1), _relu(b["Qmin"][gb] - qg).max(1)])
        M["r_volt"][sl] = np.maximum(_relu(v - b["Vmax"]).max(1), _relu(b["Vmin"] - v).max(1))
        if fin.any():
            Sm = np.maximum(np.abs(Sf), np.abs(St))[:, fin]
            M["r_br"][sl] = _relu(Sm - b["rate"][fin]).max(1)
        M["se_pg"][sl] = ((pg - YY[:, gb, 0]) ** 2).sum(1); M["se_qg"][sl] = ((qg - YY[:, gb, 1]) ** 2).sum(1)
        M["se_v"][sl] = ((v - YY[..., 2]) ** 2).sum(1)
        M["se_th"][sl] = (((yy[..., 3] - YY[..., 3]) / THETA_BASE) ** 2).sum(1)
        Js = J(YY[:, gb, 0]); M["gap"][sl] = 100.0 * np.abs(J(pg) - Js) / (np.abs(Js) + 1e-6)
    M["ng"], M["n"] = len(gb), b["n"]
    return M

def aggregate(M, mask):
    m = np.asarray(mask, bool); c = int(m.sum())
    if c == 0: return None
    r_in = np.maximum.reduce([M["r_gen"], M["r_volt"], M["r_br"]])
    ok = lambda r, t: 100.0 * float((r[m] <= t).mean())
    fe = lambda te, ti: 100.0 * float(((M["r_bal"][m] <= te) & (M["r_gen"][m] <= ti) & (M["r_volt"][m] <= ti) & (M["r_br"][m] <= ti)).mean())
    rm = lambda k, d: 1e5 * math.sqrt(M[k][m].sum() / (c * d))
    return dict(rmse_pg=rm("se_pg", M["ng"]), rmse_qg=rm("se_qg", M["ng"]), rmse_v=rm("se_v", M["n"]), rmse_th=rm("se_th", M["n"]),
                gap=float(M["gap"][m].mean()), gap_p95=float(np.percentile(M["gap"][m], 95)), gap_max=float(M["gap"][m].max()),
                r_p=float(M["r_p"][m].mean()), r_q=float(M["r_q"][m].mean()), r_bal_mean=float(M["r_bal"][m].mean()),
                feas=fe(TAU_EQ, TAU_INEQ), feas_t4=fe(1e-4, 1e-4), feas_t3=fe(1e-3, 1e-3), feas_t2=fe(1e-2, 1e-2),
                csr_bal=ok(M["r_bal"], TAU_EQ), csr_gen=ok(M["r_gen"], TAU_INEQ), csr_volt=ok(M["r_volt"], TAU_INEQ),
                csr_br=ok(M["r_br"], TAU_INEQ), max_eq=float(M["r_bal"][m].max()), max_viol=float(r_in[m].max()),
                mean_viol=float(r_in[m].mean()))

# =====================================================================================
#  EXTERNAL RECOVERY
# =====================================================================================
def project_bounds(y, b):
    y = y.copy(); gb = b["gbus"]
    y[..., 2] = np.clip(y[..., 2], b["Vmin"], b["Vmax"])
    y[..., gb, 0] = np.clip(y[..., gb, 0], b["Pmin"][gb], b["Pmax"][gb])
    y[..., gb, 1] = np.clip(y[..., gb, 1], b["Qmin"][gb], b["Qmax"][gb])
    return y

def recover_one(y, A, b, n_iter=NEWTON_IT, outer=PVPQ_OUTER):
    n, gb, sl = b["n"], b["gbus"], b["slack"]; Yb = b["Ybus"]
    y64 = project_bounds(y.astype(np.float64), b)
    A64 = A.astype(np.float64); pnet, qnet = A64[:, 0], A64[:, 1]
    gnos = np.array([g for g in gb if g != sl], dtype=int)
    Psp = -pnet.copy(); Psp[gnos] += y64[gnos, 0]
    th = y64[:, 3] - y64[sl, 3]
    Vc = y64[:, 2] * np.exp(1j * th); di = np.diag_indices(n)
    pv_act = [int(g) for g in gnos]; qfix = {}
    try:
        for _o in range(outer):
            pv = np.array(pv_act, dtype=int)
            pq = np.setdiff1d(np.arange(n), np.concatenate([pv, [sl]])).astype(int)
            pvpq = np.concatenate([pv, pq]).astype(int)
            Qsp = -qnet.copy()
            for g, qv in qfix.items(): Qsp[g] += qv
            for _it in range(n_iter):
                I = Yb @ Vc; S = Vc * np.conj(I)
                mis = np.concatenate([S.real[pvpq] - Psp[pvpq], S.imag[pq] - Qsp[pq]])
                if len(mis) == 0 or np.max(np.abs(mis)) < 1e-10: break
                Vm = np.abs(Vc); Vn = Vc / Vm
                M1 = -Yb * Vc[None, :]; M1[di] += I
                dSa = 1j * Vc[:, None] * np.conj(M1)
                dSm = Vc[:, None] * np.conj(Yb * Vn[None, :]); dSm[di] += np.conj(I) * Vn
                Jm = np.block([[dSa.real[np.ix_(pvpq, pvpq)], dSm.real[np.ix_(pvpq, pq)]],
                               [dSa.imag[np.ix_(pq, pvpq)], dSm.imag[np.ix_(pq, pq)]]])
                dx = np.linalg.solve(Jm, -mis)
                ang = np.angle(Vc); ang[pvpq] += dx[:len(pvpq)]
                Vm = Vm.copy(); Vm[pq] += dx[len(pvpq):]
                Vc = Vm * np.exp(1j * ang)
            S = Vc * np.conj(Yb @ Vc); qg = S.imag + qnet
            viol = [g for g in pv_act if qg[g] > b["Qmax"][g] + 1e-9 or qg[g] < b["Qmin"][g] - 1e-9]
            if not viol: break
            for g in viol:
                qfix[g] = float(np.clip(qg[g], b["Qmin"][g], b["Qmax"][g])); pv_act.remove(g)
        S = Vc * np.conj(Yb @ Vc)
        out = np.zeros((n, 4), np.float64)
        out[gb, 0] = (S.real + pnet)[gb]; out[gb, 1] = (S.imag + qnet)[gb]
        out[:, 2] = np.abs(Vc); out[:, 3] = np.angle(Vc)
        if not np.isfinite(out).all(): raise FloatingPointError
    except Exception:
        out = y64
    return project_bounds(out.astype(np.float32), b)

def recover_batch(Y_pred, A, b):
    return np.stack([recover_one(Y_pred[i], A[i], b) for i in range(len(Y_pred))]).astype(np.float32)

# =====================================================================================
#  DATA
# =====================================================================================
def raw_path(system, n): return f"{DATA_DIR}/{system}_acopf_generated_N{n}.npz"
def bundle_path(system): return f"{DATA_DIR}/{system}_acopf_bundle.pkl"


def perturb_load_case(base_ppc, rng, load_level=None):
    """Apply a global loading level plus buswise and correlated regional perturbations."""
    from pypower.idx_bus import PD, QD
    ppc=copy.deepcopy(base_ppc); bus=ppc["bus"]
    level=float(rng.uniform(LOAD_MIN,LOAD_MAX) if load_level is None else load_level)
    n=len(bus); region=rng.normal(0,1, size=max(3,int(np.sqrt(n))))
    regional=np.resize(region,n); regional=(regional-regional.mean())/(regional.std()+1e-9)
    independent=rng.normal(0,BUS_PERTURB_STD,n)
    mult=np.clip(level*(1+REGIONAL_CORRELATION*0.06*regional+independent),0.45,1.55)
    qmult=np.clip(mult*(1+rng.normal(0,Q_PERTURB_STD,n)),0.40,1.65)
    bus[:,PD]=base_ppc["bus"][:,PD]*mult
    bus[:,QD]=base_ppc["bus"][:,QD]*qmult
    return ppc, level, mult, qmult


def solve_acopf_scenario(base_ppc, rng):
    """Run a real nonlinear AC-OPF using PYPOWER; retry perturbed scenarios on failure."""
    from pypower.runopf import runopf
    for retry in range(OPF_MAX_RETRIES):
        case, level, mult, qmult=perturb_load_case(base_ppc,rng)
        try:
            result=runopf(case, dict(OPF_OPTIONS))
            if result and result.get("success",False) and np.isfinite(result["bus"]).all() and np.isfinite(result["gen"]).all():
                return result, level, mult, qmult, float(result.get("f",np.nan)), retry
        except Exception:
            pass
    return None


def do_data(system, force=False):
    os.makedirs(DATA_DIR,exist_ok=True); n_t=target_n(system); path=raw_path(system,n_t)
    if os.path.exists(path) and os.path.exists(bundle_path(system)) and not force:
        print(f"  cached newly-solved AC-OPF scenarios: {path}"); return
    t0=time.time(); rng=np.random.default_rng(stable_seed(SEED,system,"scenario_generation"))
    ppc=load_pypower_case(system); b,aux=build_struct(ppc)
    As,Ys,levels,objectives,perturbations,retries=[],[],[],[],[],[]
    attempt=0; max_attempts=max(n_t*3,n_t+OPF_MAX_RETRIES)
    print(f"  generating {n_t} randomized loads and solving fresh AC-OPFs with PYPOWER ...")
    while len(As)<n_t and attempt<max_attempts:
        attempt+=1; solved=solve_acopf_scenario(ppc,rng)
        if solved is None:
            if attempt%25==0: print(f"    attempts={attempt}, successful={len(As)}")
            continue
        result,level,mult,qmult,obj,retry=solved
        A,Y=sample_AY_from_solution(result,b,aux)
        if not np.isfinite(A).all() or not np.isfinite(Y).all(): continue
        As.append(A);Ys.append(Y);levels.append(level);objectives.append(obj);perturbations.append(np.stack([mult,qmult])) ;retries.append(retry)
        if len(As)%25==0 or len(As)==n_t: print(f"    OPF solved {len(As)}/{n_t} (attempt {attempt})")
    if len(As)<max(100,int(.8*n_t)):
        raise RuntimeError(f"Only {len(As)}/{n_t} OPFs converged. Reduce sample count/load range or inspect PYPOWER solver setup.")
    A,Y=np.stack(As),np.stack(Ys); alpha=A[:,:,0].sum(1); alpha=alpha/np.mean(alpha)
    M=sample_metrics(Y,A,Y,b); lab=100.*float(((M["r_bal"]<=TAU_EQ)&(np.maximum.reduce([M["r_gen"],M["r_volt"],M["r_br"]])<=TAU_INEQ)).mean())
    np.savez_compressed(path,A=A,Y=Y,alpha=alpha,load_level=np.asarray(levels),opf_objective=np.asarray(objectives),
        load_perturbations=np.asarray(perturbations),opf_retries=np.asarray(retries),label_feas=lab,attempts=attempt)
    with open(bundle_path(system),"wb") as f: pickle.dump(b,f)
    print(f"  saved {len(A)} fresh AC-OPF labels; label-feasibility={lab:.1f}%; attempts={attempt}; {time.time()-t0:.1f}s")


def make_split(N, seed):
    rng=np.random.default_rng(seed); perm=rng.permutation(N)
    n_te=min(TEST_MAX,int(round(TEST_FRAC*N))); n_va=min(VAL_MAX,int(round(VAL_FRAC*N)))
    return np.sort(perm[n_te+n_va:]),np.sort(perm[n_te:n_te+n_va]),np.sort(perm[:n_te])


def load_system(system):
    z=np.load(raw_path(system,target_n(system)))
    with open(bundle_path(system),"rb") as f: b=pickle.load(f)
    A,Y,alpha=z["A"],z["Y"],z["alpha"]
    tr,va,te=make_split(len(A),stable_seed(system,"split")); edges=np.quantile(alpha[tr],[1/3,2/3]);cats=np.digitize(alpha,edges)
    q_ex=float(np.quantile(alpha[tr],EXTRAP_Q));ex_tr=tr[alpha[tr]<=q_ex];ex_te=te[alpha[te]>q_ex]
    return SimpleNamespace(system=system,b=b,A=A,Y=Y,alpha=alpha,cats=cats,tr=tr,va=va,te=te,edges=edges,q_ex=q_ex,
        ex_tr=ex_tr,ex_te=ex_te,label_feas=float(z["label_feas"]),load_level=z["load_level"],opf_objective=z["opf_objective"],
        load_perturbations=z["load_perturbations"],opf_retries=z["opf_retries"],opf_attempts=int(z["attempts"]))

# =====================================================================================
#  MODELS
# =====================================================================================
class ProjectionLayer(nn.Module):
    def __init__(self, b, st, lo, hi):
        super().__init__()
        n, gb, sl = b["n"], b["gbus"], b["slack"]; ng = len(gb)
        ns = np.array([i for i in range(n) if i != sl], int)
        Eg = np.zeros((n, ng), np.float32); Eg[gb, np.arange(ng)] = 1.0
        Pns = np.zeros((n - 1, n), np.float32); Pns[np.arange(n - 1), ns] = 1.0
        sd = st["y_sd"][0].detach().cpu().numpy()
        w2 = np.concatenate([sd[gb, 0], sd[gb, 1], sd[:, 2], sd[ns, 3]]) ** 2
        self.n, self.ng, self.sl = n, ng, sl
        self.register_buffer("Y", torch.tensor(b["Ybus"], dtype=torch.complex64))
        self.register_buffer("Eg", torch.tensor(Eg)); self.register_buffer("Pns", torch.tensor(Pns))
        self.register_buffer("ns", torch.tensor(ns)); self.register_buffer("w2", torch.tensor(w2, dtype=torch.float32))
        self.register_buffer("eye", torch.eye(n)); self.register_buffer("eye2", torch.eye(2 * n))
        self.register_buffer("lo", lo[0].clone()); self.register_buffer("hi", hi[0].clone())
    def forward(self, y, A, K):
        B, n, ng = y.shape[0], self.n, self.ng
        for _ in range(K):
            V = torch.polar(y[..., 2].clamp(min=0.3), y[..., 3])
            I = torch.einsum("ij,bj->bi", self.Y, V); S = V * I.conj()
            h = torch.cat([y[..., 0] - A[..., 0] - S.real, y[..., 1] - A[..., 1] - S.imag], 1)
            Vn = V / V.abs().clamp(min=1e-6)
            M1 = -self.Y[None] * V[:, None, :] + I[:, :, None] * self.eye
            dSa = 1j * V[:, :, None] * M1.conj()
            dSm = V[:, :, None] * (self.Y[None] * Vn[:, None, :]).conj() + I.conj()[:, :, None] * Vn[:, None, :] * self.eye
            Eg = self.Eg[None].expand(B, -1, -1); Z = torch.zeros(B, n, ng, device=y.device)
            Jp = torch.cat([Eg, Z, -dSm.real, -dSa.real[:, :, self.ns]], 2)
            Jq = torch.cat([Z, Eg, -dSm.imag, -dSa.imag[:, :, self.ns]], 2)
            J = torch.cat([Jp, Jq], 1)
            JD = J * self.w2[None, None, :]
            Mm = JD @ J.transpose(1, 2); Mm = Mm + PROJ_EPS * (1.0 + Mm.diagonal(dim1=1, dim2=2).mean(1))[:, None, None] * self.eye2
            lam = batched_solve(Mm, h[..., None])
            dx = -(JD.transpose(1, 2) @ lam)[..., 0]
            dx = torch.nan_to_num(dx, nan=0.0, posinf=0.0, neginf=0.0)
            pg = y[..., 0] + dx[:, :ng] @ self.Eg.T
            qg = y[..., 1] + dx[:, ng:2 * ng] @ self.Eg.T
            vm = y[..., 2] + dx[:, 2 * ng:2 * ng + n]
            th = y[..., 3] + dx[:, 2 * ng + n:] @ self.Pns
            pg = torch.maximum(torch.minimum(pg, self.hi[:, 0]), self.lo[:, 0])
            qg = torch.maximum(torch.minimum(qg, self.hi[:, 1]), self.lo[:, 1])
            vm = torch.maximum(torch.minimum(vm, self.hi[:, 2]), self.lo[:, 2])
            y = torch.stack([pg, qg, vm, th], -1)
        return y

class Base(nn.Module):
    def __init__(self, st, lo, hi, bounded, proj):
        super().__init__()
        for k in ("a_mu", "a_sd", "y_mu", "y_sd", "mask"): self.register_buffer(k, st[k].clone())
        self.register_buffer("lo", lo.clone()); self.register_buffer("hi", hi.clone())
        self.bounded, self.proj, self.proj_it = bounded, proj, 0
    def norm(self, A): return (A - self.a_mu) / self.a_sd
    def finish(self, o, A):
        if self.bounded:
            rng = (self.hi - self.lo).clamp(min=1e-6)
            base = ((self.y_mu[..., :3] - self.lo) / rng).clamp(1e-3, 1 - 1e-3)
            y3 = self.lo + rng * torch.sigmoid(torch.logit(base) + o[..., :3])
            y = torch.cat([y3, self.y_mu[..., 3:] + self.y_sd[..., 3:] * o[..., 3:]], -1)
        else:
            y = self.y_mu + self.y_sd * o
        y = y * self.mask
        if self.proj is not None and self.proj_it > 0: y = self.proj(y, A, self.proj_it)
        return y

class SpectralConv(nn.Module):
    def __init__(self, w, modes):
        super().__init__(); self.modes = modes
        self.wr = nn.Parameter(torch.randn(modes, w, w, 2) * 0.5 / math.sqrt(w))
    def forward(self, x):
        n = x.shape[1]; xf = torch.fft.rfft(x, dim=1); m = min(self.modes, xf.shape[1])
        out = torch.zeros_like(xf); out[:, :m] = torch.einsum("bmi,mio->bmo", xf[:, :m], torch.view_as_complex(self.wr)[:m])
        return torch.fft.irfft(out, n=n, dim=1)

class FNOBlock(nn.Module):
    def __init__(self, w, modes):
        super().__init__(); self.pt = nn.Linear(w, w); self.spec = SpectralConv(w, modes)
    def forward(self, z): return z + F.gelu(self.pt(z) + self.spec(z))

class WaveConv(nn.Module):
    def __init__(self, w, n, levels):
        super().__init__(); self.J, self.n = levels, n; mult = 2 ** levels; self.Lp = int(math.ceil(n / mult) * mult)
        la = self.Lp // mult
        self.Wa = nn.Parameter(torch.randn(la, w, w) * 0.5 / math.sqrt(w))
        self.Wd = nn.ParameterList([nn.Parameter(torch.randn(self.Lp // 2 ** (j + 1), w, w) * 0.5 / math.sqrt(w)) for j in range(levels)])
    def forward(self, x):
        B, n, w = x.shape; r2 = 1.0 / math.sqrt(2.0)
        a = F.pad(x, (0, 0, 0, self.Lp - n)); dets = []
        for _ in range(self.J):
            e, o = a[:, 0::2], a[:, 1::2]; dets.append((e - o) * r2); a = (e + o) * r2
        a = torch.einsum("bli,lio->blo", a, self.Wa)
        dets = [torch.einsum("bli,lio->blo", d, self.Wd[j]) for j, d in enumerate(dets)]
        for j in reversed(range(self.J)):
            d = dets[j]; e, o = (a + d) * r2, (a - d) * r2
            a = torch.stack([e, o], 2).reshape(B, 2 * e.shape[1], w)
        return a[:, :n]

class WNOBlock(nn.Module):
    def __init__(self, w, n, levels):
        super().__init__(); self.pt = nn.Linear(w, w); self.wav = WaveConv(w, n, levels)
    def forward(self, z): return z + F.gelu(self.pt(z) + self.wav(z))

class SeqCore(nn.Module):
    def __init__(self, xi, blocks):
        super().__init__(); order = np.argsort(xi)
        self.register_buffer("perm", torch.tensor(order)); self.register_buffer("inv", torch.tensor(np.argsort(order)))
        self.blocks = nn.ModuleList(blocks)
    def forward(self, z):
        z = z[:, self.perm]
        for blk in self.blocks: z = blk(z)
        return z[:, self.inv]

def edge_features(b):
    ys = -b["Ybus"].copy(); np.fill_diagonal(ys, 0)
    adj = (np.abs(ys) > 1e-12).astype(np.float32); sc = float(np.abs(ys[adj > 0]).mean()) if adj.sum() > 0 else 1.0
    gd = b["gdist"].copy(); fin = np.isfinite(gd); gd[~fin] = gd[fin].max() if fin.any() else 1.0
    E = np.stack([ys.real / sc, ys.imag / sc, adj, gd / max(gd.max(), 1.0)], -1).astype(np.float32)
    deg = np.maximum(adj.sum(1, keepdims=True), 1.0)
    return E, (adj / deg).astype(np.float32)

class GNOLayer(nn.Module):
    def __init__(self, w, n_edge_feat, kw):
        super().__init__(); self.W = nn.Linear(w, w)
        self.kappa = nn.Sequential(nn.Linear(n_edge_feat, kw), nn.GELU(), nn.Linear(kw, w))
    def forward(self, h, E, adjn):
        K = self.kappa(E) * adjn[..., None]
        return h + F.gelu(self.W(h) + torch.einsum("ijw,bjw->biw", K, h))

class GNOCore(nn.Module):
    def __init__(self, b):
        super().__init__(); E, adjn = edge_features(b)
        self.register_buffer("E", torch.tensor(E)); self.register_buffer("adjn", torch.tensor(adjn))
        self.layers = nn.ModuleList([GNOLayer(W_LAT, E.shape[-1], GNO_KW) for _ in range(GNO_L)])
    def forward(self, z):
        for l in self.layers: z = l(z, self.E, self.adjn)
        return z

class OpNet(Base):
    def __init__(self, st, b, kind, lo, hi, bounded, proj):
        super().__init__(st, lo, hi, bounded, proj); n = b["n"]
        self.lift = nn.Linear(N_CH, W_LAT); self.emb = nn.Parameter(torch.randn(1, n, W_LAT) * 0.1)
        if kind == "fno": self.core = SeqCore(b["xi"], [FNOBlock(W_LAT, min(M_MODES, n // 2 + 1)) for _ in range(L_BLK)])
        elif kind == "wno": self.core = SeqCore(b["xi"], [WNOBlock(W_LAT, n, WNO_LEVELS) for _ in range(L_BLK)])
        else: self.core = GNOCore(b)
        self.head = nn.Sequential(nn.Linear(W_LAT, W_LAT), nn.GELU(), nn.Linear(W_LAT, 4))
    def forward(self, A):
        z = self.core(self.lift(self.norm(A)) + self.emb)
        return self.finish(self.head(z), A)

class DeepONet(Base):
    def __init__(self, st, b, lo, hi, bounded, proj):
        super().__init__(st, lo, hi, bounded, proj); n = b["n"]; self.n = n
        self.register_buffer("coords", torch.tensor(np.stack([b["xi"], b["xi2"]], 1), dtype=torch.float32))
        self.branch = nn.Sequential(nn.Linear(n * N_CH, DON_BRANCH_W), nn.GELU(), nn.Linear(DON_BRANCH_W, DON_BRANCH_W), nn.GELU(),
                                    nn.Linear(DON_BRANCH_W, 4 * DON_P))
        self.trunk = nn.Sequential(nn.Linear(2, DON_TRUNK_W), nn.GELU(), nn.Linear(DON_TRUNK_W, DON_TRUNK_W), nn.GELU(), nn.Linear(DON_TRUNK_W, DON_P))
        self.bias = nn.Parameter(torch.zeros(1, n, 4))
    def forward(self, A):
        Bk = self.branch(self.norm(A).flatten(1)).view(-1, 4, DON_P); T = self.trunk(self.coords)
        o = torch.einsum("bcp,np->bnc", Bk, T) / math.sqrt(DON_P) + self.bias
        return self.finish(o, A)

class Dense(Base):
    def __init__(self, st, n, lo, hi):
        super().__init__(st, lo, hi, False, None); self.n = n
        layers, d = [], n * N_CH
        for _ in range(MLP_L): layers += [nn.Linear(d, MLP_W), nn.ReLU()]; d = MLP_W
        layers += [nn.Linear(d, n * 4)]; self.net = nn.Sequential(*layers)
    def forward(self, A):
        return self.finish(self.net(self.norm(A).flatten(1)).view(-1, self.n, 4), A)

class PhysLoss:
    def __init__(self, b, st):
        t = lambda a, dt=torch.float32: torch.tensor(np.asarray(a), dtype=dt, device=DEVICE)
        self.Y, self.Yf, self.Yt = (t(b[k], torch.complex64) for k in ("Ybus", "Yf", "Yt"))
        self.rate = t(np.minimum(b["rate"], 1e6)); self.gb = torch.tensor(b["gbus"], device=DEVICE)
        self.fb = torch.tensor(b["fb"], device=DEVICE); self.tb = torch.tensor(b["tb"], device=DEVICE)
        g = b["gbus"]
        self.pl, self.ph, self.ql, self.qh = (t(b[k][g]) for k in ("Pmin", "Pmax", "Qmin", "Qmax"))
        self.vl, self.vh = t(b["Vmin"]), t(b["Vmax"]); self.ysd = st["y_sd"]; self.m = st["mask"]
        self.c2, self.c1, self.c0 = t(b["c2"]), t(b["c1"]), t(b["c0"]); self.sn = b["sn"]; self.cs = st["cost_scale"]
    def comps(self, y, A, ys, lam):
        o = {}
        if lam["sup"] > 0: o["sup"] = (((y - ys) / self.ysd) ** 2 * self.m).sum((1, 2)) / self.m.sum()
        pg, qg, v = y[:, self.gb, 0], y[:, self.gb, 1], y[..., 2]
        if lam["cost"] > 0: o["cost"] = (self.c2 * (pg * self.sn) ** 2 + self.c1 * pg * self.sn + self.c0).sum(1) / self.cs
        need_v = lam["bal"] > 0 or lam["flow"] > 0
        if need_v:
            V = torch.polar(y[..., 2].clamp(min=1e-3), y[..., 3])
            if lam["bal"] > 0:
                S = V * torch.conj(torch.einsum("ij,bj->bi", self.Y, V))
                o["bal"] = ((y[..., 0] - A[..., 0] - S.real) ** 2 + (y[..., 1] - A[..., 1] - S.imag) ** 2).sum(1)
            if lam["flow"] > 0:
                If, It = torch.einsum("ij,bj->bi", self.Yf, V), torch.einsum("ij,bj->bi", self.Yt, V)
                Sm = torch.maximum((V[:, self.fb] * If.conj()).abs(), (V[:, self.tb] * It.conj()).abs())
                o["flow"] = (F.relu(Sm - self.rate) ** 2).sum(1)
        if lam["ineq"] > 0:
            o["ineq"] = (F.relu(pg - self.ph) ** 2 + F.relu(self.pl - pg) ** 2 + F.relu(qg - self.qh) ** 2 + F.relu(self.ql - qg) ** 2).sum(1) \
                      + (F.relu(v - self.vh) ** 2 + F.relu(self.vl - v) ** 2).sum(1)
        return o
    def per_sample(self, y, A, ys, lam):
        c = self.comps(y, A, ys, lam)
        return sum(lam[k] * x for k, x in c.items())

def make_stats(ctx, tr, labels):
    b, n, gb = ctx.b, ctx.b["n"], ctx.b["gbus"]; A = ctx.A[tr]
    a_sd = A.std(0); a_sd = np.where(a_sd < 1e-6, 1.0, a_sd)
    mask = np.ones((1, n, 4), np.float32); mask[0, :, 0] = b["chi_g"]; mask[0, :, 1] = b["chi_g"]; mask[0, b["slack"], 3] = 0.0
    Pmin, Pmax, Qmin, Qmax, Vmin, Vmax = (b[k] for k in ("Pmin", "Pmax", "Qmin", "Qmax", "Vmin", "Vmax"))
    pd_tot = A[:, :, 0].sum(1)
    frac = np.clip((pd_tot - Pmin.sum()) / max(Pmax.sum() - Pmin.sum(), 1e-9), 0, 1)
    pg_s = Pmin[None, :] + frac[:, None] * (Pmax - Pmin)[None, :]
    cost0 = (b["c2"] * (pg_s[:, gb] * b["sn"]) ** 2 + b["c1"] * pg_s[:, gb] * b["sn"] + b["c0"]).sum(1)
    cost_scale = max(float(np.mean(np.abs(cost0))), 1e-6)
    if labels:
        Y = ctx.Y[tr]; y_mu = Y.mean(0) * mask[0]; y_sd = np.maximum(Y.std(0), 1e-3)
    else:
        y_mu = np.zeros((n, 4)); y_mu[:, 0] = pg_s.mean(0); y_mu[:, 1] = 0.5 * (Qmin + Qmax); y_mu[:, 2] = np.clip(1.0, Vmin, Vmax); y_mu *= mask[0]
        y_sd = np.stack([np.maximum(0.25 * (Pmax - Pmin), 1e-2), np.maximum(0.25 * (Qmax - Qmin), 1e-2),
                         np.maximum(0.25 * (Vmax - Vmin), 1e-3), np.full(n, THETA_SCALE)], 1)
    T = lambda a: torch.tensor(np.asarray(a, np.float32), device=DEVICE)
    return dict(a_mu=T(A.mean(0))[None], a_sd=T(a_sd)[None], y_mu=T(y_mu)[None], y_sd=T(y_sd)[None], mask=T(mask), cost_scale=cost_scale)

def box_bounds(ctx):
    b = ctx.b; g = b["chi_g"]
    L_ = np.stack([b["Pmin"] * g, b["Qmin"] * g, b["Vmin"]], 1); H_ = np.stack([b["Pmax"] * g, b["Qmax"] * g, b["Vmax"]], 1)
    H_ = np.maximum(H_, L_ + 1e-3 * np.stack([g, g, np.ones_like(g)], 1))
    return (torch.tensor(L_[None], dtype=torch.float32, device=DEVICE), torch.tensor(H_[None], dtype=torch.float32, device=DEVICE))

def build_model(ctx, cfg, st):
    b = ctx.b; lo, hi = box_bounds(ctx); arch = cfg["arch"]
    proj = ProjectionLayer(b, st, lo, hi).to(DEVICE) if cfg["proj_it"] > 0 else None
    if arch in ("fno", "wno", "gno"): m = OpNet(st, b, arch, lo, hi, cfg["bounded"], proj)
    elif arch == "don": m = DeepONet(st, b, lo, hi, cfg["bounded"], proj)
    else: m = Dense(st, b["n"], lo, hi)
    m = m.to(DEVICE); m.proj_it = cfg["proj_it"]
    return m

@torch.no_grad()
def predict(model, A, bs=256):
    model.eval(); out = []
    for s in range(0, len(A), bs):
        out.append(model(torch.tensor(A[s:s + bs], device=DEVICE)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)

@torch.no_grad()
def measure_latency(model, A, reps=3):
    model.eval(); At = torch.tensor(A[:256], device=DEVICE)
    sync = (lambda: torch.cuda.synchronize()) if DEVICE.type == "cuda" else (lambda: None)
    model(At[:64]); sync(); ts = []
    for _ in range(reps):
        t = time.perf_counter(); model(At); sync(); ts.append((time.perf_counter() - t) / len(At) * 1e3)
    one = []
    for i in range(min(40, len(At))):
        t = time.perf_counter(); model(At[i:i + 1]); sync(); one.append((time.perf_counter() - t) * 1e3)
    return dict(b256=float(min(ts)), b1=float(np.median(one)))

def train_model(ctx, name, cfg, seed, tr_idx=None, verbose=True):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    tr = ctx.tr if tr_idx is None else tr_idx; va = ctx.va; lab = cfg["labels"]
    st = make_stats(ctx, tr, lab); model = build_model(ctx, cfg, st); lam = cfg["lam"]; pl = PhysLoss(ctx.b, st)
    final_K = model.proj_it
    if not cfg.get("proj_train", True): model.proj_it = 0
    T = lambda a: torch.tensor(a, device=DEVICE)
    At, Av = T(ctx.A[tr]), T(ctx.A[va]); Yt = T(ctx.Y[tr]) if lab else None; Yv = T(ctx.Y[va]) if lab else None
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR_MIN)
    B = min(BATCH, len(tr)); nb = max(int(math.ceil(len(tr) / B)), MIN_STEPS)
    best, bad, best_state, t0 = float("inf"), 0, None, time.time(); ep = 0
    for ep in range(EPOCHS):
        model.train()
        for _ in range(nb):
            ii = torch.tensor(rng.choice(len(tr), B, replace=False), device=DEVICE); A_ = At[ii]
            loss = pl.per_sample(model(A_), A_, Yt[ii] if lab else None, lam).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), CLIP); opt.step()
        sch.step(); model.eval()
        with torch.no_grad():
            vs = [float(pl.per_sample(model(Av[s:s + 512]), Av[s:s + 512], Yv[s:s + 512] if lab else None, lam).mean())
                  * len(Av[s:s + 512]) for s in range(0, len(Av), 512)]
            v = sum(vs) / len(Av)
        if not math.isfinite(v): break
        if v < best - 1e-12:
            if v < best * (1 - MIN_REL_GAIN) or best_state is None: bad = 0
            else: bad += 1
            best, best_state = v, {k: x.clone() for k, x in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE and ep > EARLY_STOP_AFTER * EPOCHS: break
        if verbose and ((ep + 1) % 25 == 0 or ep == 0): print(f"      [{name}] ep {ep+1:>3}/{EPOCHS} val={v:.3e} best={best:.3e} ({time.time()-t0:.0f}s)")
    if best_state is not None: model.load_state_dict(best_state)
    model.proj_it = final_K
    model.info = dict(epochs=ep + 1, best=float(best), secs=time.time() - t0, early=bool(bad >= PATIENCE),
                      params=int(sum(p.numel() for p in model.parameters())), n_train=int(len(tr)))
    return model.eval()

# =====================================================================================
#  EXPERIMENTS
# =====================================================================================
def eval_pred(ctx, model, idx):
    A, Y = ctx.A[idx], ctx.Y[idx]
    y_raw = predict(model, A); y_rec = recover_batch(y_raw, A, ctx.b)
    out = dict(raw=sample_metrics(y_raw, A, Y, ctx.b), rec=sample_metrics(y_rec, A, Y, ctx.b))
    if model.proj is not None and model.proj_it > 0:
        k = model.proj_it; model.proj_it = 0; out["pre"] = sample_metrics(predict(model, A), A, Y, ctx.b); model.proj_it = k
    else: out["pre"] = out["raw"]
    return out

def run_system(ctx, n_seeds, abl_seeds):
    te = ctx.te
    out = SimpleNamespace(system=ctx.system, n=ctx.b["n"], main={m: [] for m in ORDER}, extrap={m: [] for m in ORDER}, cats_te=ctx.cats[te],
                          abl=None, sweep={}, lat={m: [] for m in ORDER}, ref=[dict(raw=sample_metrics(ctx.Y[te], ctx.A[te], ctx.Y[te], ctx.b))],
                          info=dict(N=len(ctx.A), tr=len(ctx.tr), va=len(ctx.va), te=len(te), label_feas=ctx.label_feas,
                                    edges=np.round(ctx.edges, 4).tolist(), q_ex=float(ctx.q_ex), ex_tr=len(ctx.ex_tr), ex_te=len(ctx.ex_te),
                                    cat_te=[int((ctx.cats[te] == c).sum()) for c in range(3)],
                                    cat_tr=[int((ctx.cats[ctx.tr] == c).sum()) for c in range(3)],
                                    alpha_rng=[float(ctx.alpha.min()), float(ctx.alpha.max())]))
    out.ex_mask = np.isin(te, ctx.ex_te); out.train_info = {m: [] for m in ORDER}; out.extrap_info = {m: [] for m in ORDER}; out.abl_info = {}
    try: out.supp = collect_supp_data(ctx)
    except Exception as e: out.supp = None; print(f"  WARNING: supplementary data collection failed ({type(e).__name__}: {e})")
    t = time.perf_counter(); recover_batch(ctx.Y[te[:20]], ctx.A[te[:20]], ctx.b); out.t_newton = (time.perf_counter() - t) / min(20, len(te)) * 1e3
    for seed in range(n_seeds):
        for name in ORDER:
            print(f"    seed {seed} | main training: {name}")
            m = train_model(ctx, name, MODEL_CFG[name], stable_seed(ctx.system, name, seed, "main"), verbose=(seed == 0))
            out.main[name].append(eval_pred(ctx, m, te)); out.train_info[name].append(m.info); out.lat[name].append(measure_latency(m, ctx.A[te]))
            if name == "PNO-FNO":
                for K in SWEEP_K:
                    k0 = m.proj_it; m.proj_it = K
                    out.sweep.setdefault(K, []).append(sample_metrics(predict(m, ctx.A[te]), ctx.A[te], ctx.Y[te], ctx.b)); m.proj_it = k0
    if len(ctx.ex_te) >= 20 and len(ctx.ex_tr) >= 50:
        for seed in range(EXTRAP_SEEDS):
            for name in ORDER:
                print(f"    seed {seed} | load-extrapolation training (loading <= q{int(EXTRAP_Q*100)}): {name}")
                m = train_model(ctx, name, MODEL_CFG[name], stable_seed(ctx.system, name, seed, "extrap"), tr_idx=ctx.ex_tr, verbose=False)
                out.extrap[name].append(eval_pred(ctx, m, ctx.ex_te)); out.extrap_info[name].append(m.info)
    if ctx.system == ABL_SYSTEM:
        out.abl = {"Full PNO-FNO": [out.main["PNO-FNO"][0]]}
        for label, ov in ABLATIONS:
            cfg = copy.deepcopy(MODEL_CFG["PNO-FNO"]); cfg.update(ov); runs = []
            for s in range(abl_seeds):
                print(f"    ablation '{label}' seed {s}")
                m = train_model(ctx, "PNO-FNO", cfg, stable_seed(ctx.system, label, s), verbose=False)
                runs.append(eval_pred(ctx, m, te)); out.abl_info.setdefault(label, []).append(m.info)
            out.abl[label] = runs
    return out

# =====================================================================================
#  TABLES
# =====================================================================================
OUT = []
def emit(s=""): print(s); OUT.append(s)

def S(vals, p=2, sci=False):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v: return "--"
    f = (lambda x: f"{x:.{p}e}") if sci else (lambda x: f"{x:.{p}f}")
    return f(np.mean(v)) if len(v) == 1 else f"{f(np.mean(v))} ± {f(np.std(v))}"

def table(name, title, header, rows, note=None):
    body = [[str(c) for c in r] for r in rows if not isinstance(r, str)]
    w = [max(len(h), *(len(r[i]) for r in body)) if body else len(h) for i, h in enumerate(header)]
    ln = lambda cells: "  ".join(c.ljust(w[i]) if i < 2 else c.rjust(w[i]) for i, c in enumerate(cells))
    emit(f"\n{title}"); emit(ln(header)); emit("-" * (sum(w) + 2 * (len(w) - 1)))
    for r in rows: emit(f"[{r}]" if isinstance(r, str) else ln([str(c) for c in r]))
    if note: emit(f"note: {note}")
    os.makedirs(RES_DIR, exist_ok=True)
    with open(f"{RES_DIR}/{name}.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(header); [wr.writerow(r) for r in body]

def sv(runs, variant, mask, key):
    o = [aggregate(r[variant], mask) for r in runs]; return [a[key] for a in o if a is not None]

def table_related():
    X, C, P = "×", "✓", "Partial"
    rows = [["Supervised deep-learning surrogates", X, P, X, X, "Donti2021, Pan2020"],
            ["Unsupervised PINN with soft penalties", C, X, X, X, "Nellikkath2021"],
            ["Augmented-Lagrangian training", C, P, X, X, "article Sec. 2.2"],
            ["Neural operators (FNO / DeepONet / GNO / WNO)", C, X, C, P, "Li2021FNO, Lu2021DeepONet, Li2020GNO, Tripura2023WNO"],
            ["Proposed Unified PNO framework", C, C, C, C, "This work"]]
    table("table_related_work", "Table A (static, from the article). Qualitative comparison of learning-based OPF directions",
          ["Method class", "Refs", "Label-free", "Strict feasibility", "Global coupling", "Backbone-agnostic"],
          [[r[0], r[5], r[1], r[2], r[3], r[4]] for r in rows])

def table_comparison(Rs):
    rows = []
    for R in Rs:
        rows.append(R.system); allm = np.ones(len(R.cats_te), bool)
        for m in ["AC-OPF", "Sup-MLP", "Unsup-FNO-Soft"] + NET:
            if m == "AC-OPF":
                rr = R.ref; lat = "--"
            else:
                rr = R.main[m]; lat = S([x["b256"] for x in R.lat[m]], 2)
            rows.append([R.system, DISPLAY[m], LABELS[m], S(sv(rr, "raw", allm, "gap"), 2), S(sv(rr, "raw", allm, "r_p"), 2, True),
                         S(sv(rr, "raw", allm, "feas"), 1), lat])
    table("table1_comparison", "Table 1. Performance comparison of the PNO framework across four neural-operator backbones (out-of-sample test set)",
          ["System", "Method", "Labels?", "Cost Gap (%)", "r_P (p.u.)", "Feas (%)", "Latency (ms)"], rows,
          note=f"strict feasibility: max(|r_P|,|r_Q|)<={TAU_EQ:g}, box and thermal violations<={TAU_INEQ:g} p.u.")

def table_protocol():
    rows = [["Framework", "PyTorch"], ["Data", "Fresh randomized-load AC-OPF scenarios solved with PYPOWER: " + ", ".join(CASES[s] for s in SYSTEMS)],
            ["Labels used for training", "none for the PNO models and the soft-penalty FNO (labels only for the supervised MLP and for evaluation)"],
            ["Data split", f"train / val / test with test <= {TEST_MAX} scenarios (random, fixed seed)"], ["Batch size", BATCH],
            ["Optimizer", "Adam"], ["Initial LR", f"{LR:g}"], ["LR schedule", f"cosine to {LR_MIN:g}"], ["Max epochs", EPOCHS],
            ["Early stopping", f"{PATIENCE} epochs without improvement, only after {int(EARLY_STOP_AFTER*100)}% of epochs"],
            ["Min steps / epoch", MIN_STEPS], ["Weight decay", f"{WD:g}"], ["Gradient clipping", f"max norm {CLIP}"],
            ["Latent width / blocks", f"{W_LAT} / {L_BLK}"], ["FNO", f"spectral conv., {M_MODES} modes, Fiedler-ordered buses, residual blocks"],
            ["DeepONet", f"branch {DON_BRANCH_W}x2 -> {DON_P} basis functions, trunk {DON_TRUNK_W}x2 on 2-D bus coordinates"],
            ["GNO", f"{GNO_L} kernel message-passing layers, edge features (Re/Im admittance, adjacency, hop distance), kernel width {GNO_KW}"],
            ["WNO", f"Haar DWT, {WNO_LEVELS} levels, per-coefficient channel mixing"],
            ["Bound-preserving output", "lo + sigmoid(.)(hi-lo) for Pg, Qg, |V| (output bias initialised at a label-free proportional dispatch)"],
            ["Projection layer", f"K={PROJ_IT} unrolled weighted pseudo-inverse Newton steps (damping {PROJ_EPS:g}); training and inference"],
            ["PNO loss weights", f"cost={LAM_PNO['cost']}, balance={LAM_PNO['bal']}, thermal={LAM_PNO['flow']}"],
            ["Soft-FNO loss weights", f"cost={LAM_SOFT['cost']}, balance={LAM_SOFT['bal']}, thermal={LAM_SOFT['flow']}, box={LAM_SOFT['ineq']}"],
            ["Supervised MLP", f"{MLP_L}x{MLP_W}, MSE to the newly solved AC-OPF labels"],
            ["External recovery (comparison only)", f"Pi_phys: bound projection + Newton (<= {NEWTON_IT} its) with PV->PQ switching (<= {PVPQ_OUTER} rounds)"],
            ["Models (order)", ", ".join(ORDER)], ["Independent runs", N_SEEDS], ["Device", str(DEVICE)]]
    table("table2_training_protocol", "Table 2. Training protocol and PNO configuration", ["Item", "Setting"], rows)

def table_dataset(Rs):
    rows = []
    for R in Rs:
        i = R.info
        rows.append([R.system, R.n, i["N"], f"{i['tr']}/{i['va']}/{i['te']}", "/".join(map(str, i["cat_tr"])), "/".join(map(str, i["cat_te"])),
                     f"{i['alpha_rng'][0]:.3f}-{i['alpha_rng'][1]:.3f}", f"{i['label_feas']:.1f}"])
    table("table0_dataset", "Table 0 (extra). Dataset summary", ["System", "Buses", "Scenarios", "Train/Val/Test", "Train Light/Nom/Heavy",
          "Test Light/Nom/Heavy", "alpha_sys range", "Label feas.(%)"], rows)

def table_feas_detail(Rs):
    for variant, tag in (("raw", "model output as deployed (PNO: after its projection layer)"), ("pre", "before the projection layer (bounded network output)")):
        rows = []
        for R in Rs:
            rows.append(R.system); allm = np.ones(len(R.cats_te), bool)
            for m in ["AC-OPF"] + ORDER:
                if m == "AC-OPF" and variant == "pre": continue
                r = R.ref if m == "AC-OPF" else R.main[m]
                rows.append([R.system, DISPLAY[m]] + [S(sv(r, variant, allm, k), 1) for k in ("feas", "csr_bal", "csr_gen", "csr_volt", "csr_br")] +
                            [S(sv(r, variant, allm, "r_p"), 2, True), S(sv(r, variant, allm, "r_q"), 2, True),
                             S(sv(r, variant, allm, "max_eq"), 1, True), S(sv(r, variant, allm, "max_viol"), 3)])
        table(f"table3_feasibility_{variant}", f"Table 3. Feasibility and constraint-violation detail - {tag}",
              ["System", "Method", "Strict(%)", "Balance(%)", "GenBox(%)", "Voltage(%)", "Thermal(%)", "r_P mean", "r_Q mean", "Max eq.resid", "Max viol."], rows)

def table_feas_tol(Rs):
    rows = []
    for R in Rs:
        rows.append(R.system); allm = np.ones(len(R.cats_te), bool)
        for m in ["AC-OPF"] + ORDER:
            r = R.ref if m == "AC-OPF" else R.main[m]
            rows.append([R.system, DISPLAY[m]] + [S(sv(r, v, allm, k), 1) for v in ("raw", "rec") if not (m == "AC-OPF" and v == "rec")
                                                  for k in ("feas", "feas_t4", "feas_t3", "feas_t2")] + (["--"] * 4 if m == "AC-OPF" else []))
    table("table3b_feas_tolerance", "Table 3b. Feasibility (%) at four tolerance settings (p.u.)",
          ["System", "Method", "raw strict", "raw 1e-4", "raw 1e-3", "raw 1e-2", "rec strict", "rec 1e-4", "rec 1e-3", "rec 1e-2"], rows)

def table_accuracy(Rs):
    rows = []
    for R in Rs:
        rows.append(R.system); allm = np.ones(len(R.cats_te), bool)
        for m in ORDER:
            r = R.main[m]
            rows.append([R.system, DISPLAY[m]] + [S(sv(r, "raw", allm, k), 1) for k in ("rmse_pg", "rmse_qg", "rmse_v", "rmse_th")] +
                        [S(sv(r, "raw", allm, "gap"), 3), S(sv(r, "raw", allm, "gap_p95"), 3), S(sv(r, "raw", allm, "gap_max"), 2)])
    table("table4_accuracy", "Table 4. Accuracy against fresh AC-OPF solutions (RMSE in 1e-5 p.u., mean ± std over runs)",
          ["System", "Method", "RMSE pg", "RMSE qg", "RMSE v", "RMSE theta", "Gap mean(%)", "Gap p95(%)", "Gap max(%)"], rows)

def table_ablation(Rs):
    rows = []
    for R in Rs:
        if not R.abl: continue
        allm = np.ones(len(R.cats_te), bool); rows.append(R.system)
        for nm, runs in R.abl.items():
            rows.append([R.system, nm, S(sv(runs, "raw", allm, "gap"), 3), S(sv(runs, "raw", allm, "r_p"), 2, True), S(sv(runs, "pre", allm, "r_p"), 2, True),
                         S(sv(runs, "raw", allm, "feas"), 1), S(sv(runs, "rec", allm, "feas"), 1), S(sv(runs, "raw", allm, "csr_gen"), 1),
                         S(sv(runs, "raw", allm, "csr_volt"), 1), S(sv(runs, "raw", allm, "max_viol"), 3)])
    table("table5_ablation", "Table 5. Ablation of the PNO-FNO components",
          ["System", "Variant", "Gap(%)", "r_P final", "r_P pre-proj.", "Feas raw", "Feas rec", "GenBox(%)", "Voltage(%)", "Max viol."], rows)

def table_levels(Rs):
    rows = []
    for R in Rs:
        rows.append(R.system)
        for m in ["Unsup-FNO-Soft", "Sup-MLP"] + NET:
            for c in range(3):
                mk = R.cats_te == c
                if not mk.any(): continue
                r = R.main[m]
                rows.append([R.system, f"{DISPLAY[m]} / {LEVEL_NAMES[c]}", S(sv(r, "raw", mk, "gap"), 3), S(sv(r, "raw", mk, "r_p"), 2, True),
                             S(sv(r, "raw", mk, "feas"), 1), S(sv(r, "rec", mk, "feas"), 1), S(sv(r, "raw", mk, "max_viol"), 3), int(mk.sum())])
    table("table6_loading_levels", "Table 6. Performance per macro-loading level alpha_sys (training-set terciles)",
          ["System", "Method / loading level", "Gap(%)", "r_P", "Feas raw(%)", "Feas rec(%)", "Max viol.", "n test"], rows)

def table_extrap(Rs):
    rows = []
    for R in Rs:
        if not any(R.extrap[m] for m in ORDER): continue
        rows.append(R.system); mk_main = R.ex_mask
        for m in ORDER:
            if not R.extrap[m] or not mk_main.any(): continue
            allx = np.ones(int(len(R.extrap[m][0]["raw"]["gap"])), bool)
            rows.append([R.system, DISPLAY[m], S(sv(R.main[m], "raw", mk_main, "feas"), 1), S(sv(R.extrap[m], "raw", allx, "feas"), 1),
                         S(sv(R.main[m], "raw", mk_main, "gap"), 3), S(sv(R.extrap[m], "raw", allx, "gap"), 3),
                         S(sv(R.main[m], "raw", mk_main, "r_p"), 2, True), S(sv(R.extrap[m], "raw", allx, "r_p"), 2, True)])
    table("table7_extrapolation", f"Table 7. Load extrapolation: trained on loading <= q{int(EXTRAP_Q*100)}, tested on the heaviest {int(round((1-EXTRAP_Q)*100))}% of loadings",
          ["System", "Method", "Feas in-dist.", "Feas extrap.", "Gap in-dist.", "Gap extrap.", "r_P in-dist.", "r_P extrap."], rows)

def print_all(Rs):
    OUT.clear(); emit_header(Rs)
    table_related(); table_comparison(Rs); table_protocol(); table_dataset(Rs); table_feas_detail(Rs); table_feas_tol(Rs); table_accuracy(Rs)
    table_ablation(Rs); table_levels(Rs); table_extrap(Rs)
    print_supplementary(Rs)
    os.makedirs(RES_DIR, exist_ok=True)
    with open(f"{RES_DIR}/tables.txt", "w", encoding="utf-8") as f: f.write("\n".join(OUT))
    print(f"\nSaved ALL tables in ONE file: {os.path.abspath(RES_DIR + '/tables.txt')}  (+ one CSV per table in {RES_DIR}/)")

# =====================================================================================
#  SUPPLEMENTARY TABLES
# =====================================================================================
def _arr(M, key): return np.maximum.reduce([M["r_gen"], M["r_volt"], M["r_br"]]) if key == "r_in" else M[key]
def _mean(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")
def _f(x, p=2, sci=False):
    if x is None or not np.isfinite(x): return "--"
    return f"{x:.{p}e}" if sci else f"{x:.{p}f}"
def _allm(R): return np.ones(len(R.cats_te), bool)
def _mm(runs, variant, mask, key): return _mean(sv(runs, variant, mask, key))

def _pval(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(a) != len(b) or np.allclose(a, b): return float("nan")
    try: return float(sst.ttest_rel(a, b).pvalue)
    except Exception: return float("nan")

def _winrate(runs_a, runs_b, variant, mask, fn):
    w = []
    for a, b in zip(runs_a, runs_b):
        x, y = fn(a[variant])[mask], fn(b[variant])[mask]
        w.append(100.0 * ((x < y).mean() + 0.5 * (x == y).mean()))
    return _mean(w)

def collect_supp_data(ctx):
    b, d = ctx.b, {}
    key = np.round(ctx.A[..., :2].reshape(len(ctx.A), -1) * 1e5).astype(np.int64)
    kt = set(map(bytes, key[ctx.tr])); dup = int(sum(bytes(k) in kt for k in key[ctx.te]))
    d["leak"] = dict(trva=len(set(ctx.tr) & set(ctx.va)), trte=len(set(ctx.tr) & set(ctx.te)), vate=len(set(ctx.va) & set(ctx.te)),
                     unique=int(len(np.unique(key, axis=0))), dup_trte=dup)
    d["struct"] = dict(n_branch=int(b["Yf"].shape[0]), n_gen_bus=int(len(b["gbus"])), n_rated=int((b["rate"] < 1e5).sum()),
                       requested=int(target_n(ctx.system)))
    return d

def st_env(Rs):
    dev = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU"
    rows = [
        ["Generated", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Command line", " ".join(sys.argv)],
        ["Python / platform", f"{sys.version.split()[0]} / {platform.platform()}"],
        ["torch / numpy / scipy / pandas", f"{torch.__version__} / {np.__version__} / {scipy.__version__} / {pd.__version__}"],
        ["PYPOWER", "fresh nonlinear AC-OPF labels (runopf)"],
        ["Scenario loading / perturbation", f"global level {LOAD_MIN:.0%}-{LOAD_MAX:.0%}; buswise std={BUS_PERTURB_STD:.2f}; Q std={Q_PERTURB_STD:.2f}"],
        ["Compute device / CPU cores", f"{dev} / {os.cpu_count()}"],
        ["Networks (PYPOWER/MATPOWER cases)", ", ".join(f"{s}={CASES[s]}" for s in SYSTEMS)],
        ["Scenarios per system (this run)", ", ".join(f"{s}={target_n(s)}" for s in SYSTEMS)],
        ["Independent seeds / ablation seeds / extrapolation seeds / ablation system", f"{N_SEEDS} / {ABL_SEEDS} / {EXTRAP_SEEDS} / {ABL_SYSTEM}"],
        ["Max epochs / patience / min steps / early-stop-after", f"{EPOCHS} / {PATIENCE} / {MIN_STEPS} / {EARLY_STOP_AFTER}"],
        ["Batch / LR / LR min / weight decay / grad-clip", f"{BATCH} / {LR:g} / {LR_MIN:g} / {WD:g} / {CLIP}"],
        ["PNO / soft-FNO loss coefficients", f"{LAM_PNO} / {LAM_SOFT}"],
        ["Width / blocks / FNO modes / WNO levels", f"{W_LAT} / {L_BLK} / {M_MODES} / {WNO_LEVELS}"],
        ["DeepONet basis / trunk width / branch width; GNO layers / kernel width", f"{DON_P} / {DON_TRUNK_W} / {DON_BRANCH_W}; {GNO_L} / {GNO_KW}"],
        ["Projection iterations / damping / theta scale", f"{PROJ_IT} / {PROJ_EPS:g} / {THETA_SCALE}"],
        ["Tolerances tau_eq / tau_ineq (p.u.)", f"{TAU_EQ:g} / {TAU_INEQ:g}"],
        ["External Newton iterations / PV-PQ rounds", f"{NEWTON_IT} / {PVPQ_OUTER}"],
        ["Generator costs in MW units / base MVA", f"{COST_IN_MW} / {BASE_MVA}"],
        ["Global random seed", SEED],
    ]
    return ["Item", "Value"], rows, "Everything needed to reproduce this run."

def st_integrity(Rs):
    Rs = [R for R in Rs if getattr(R, "supp", None)]
    items = [("Buses", lambda R: R.n), ("Branches", lambda R: R.supp["struct"]["n_branch"]), ("Generator buses", lambda R: R.supp["struct"]["n_gen_bus"]),
             ("Branches with thermal rating", lambda R: R.supp["struct"]["n_rated"]),
             ("Scenarios requested", lambda R: R.supp["struct"]["requested"]), ("Scenarios kept", lambda R: R.info["N"]), ("OPF attempts", lambda R: getattr(R, "opf_attempts", "n/a")), ("Mean OPF retries", lambda R: float(np.mean(getattr(R, "opf_retries", [0])))),
             ("Unique load vectors", lambda R: R.supp["leak"]["unique"]),
             ("Train / Val / Test", lambda R: f"{R.info['tr']}/{R.info['va']}/{R.info['te']}"),
             ("Index overlap train-val", lambda R: R.supp["leak"]["trva"]), ("Index overlap train-test", lambda R: R.supp["leak"]["trte"]),
             ("Index overlap val-test", lambda R: R.supp["leak"]["vate"]),
             ("Test load vectors that also occur in train", lambda R: R.supp["leak"]["dup_trte"]),
             ("Reference-label feasibility, article tolerances (%)", lambda R: f"{R.info['label_feas']:.1f}"),
             ("alpha_sys tercile edges (train)", lambda R: str(R.info["edges"])),
             ("Extrapolation: quantile / train n / test n", lambda R: f"{R.info['q_ex']:.3f} / {R.info['ex_tr']} / {R.info['ex_te']}")]
    rows = [[nm, ""] + [fn(R) for R in Rs] for nm, fn in items]
    return ["Item", ""] + [R.system for R in Rs], rows, "Overlaps must be 0 (no leakage between splits)."

def st_training(Rs):
    rows = []
    for R in Rs:
        ti = getattr(R, "train_info", None)
        if not ti: continue
        rows.append(R.system)
        for tag, src in ((" [main]", ti), (" [extrap]", getattr(R, "extrap_info", None) or {})):
            for m in ORDER:
                Lh = src.get(m, [])
                if not Lh: continue
                ep = [x["epochs"] for x in Lh]; sec = [x["secs"] for x in Lh]
                rows.append([R.system, DISPLAY[m] + tag, f"{Lh[0]['params']:,}", S(ep, 1), f"{sum(bool(x['early']) for x in Lh)}/{len(Lh)}",
                             S([x["best"] for x in Lh], 2, True), S(sec, 1), S([a / max(b_, 1) for a, b_ in zip(sec, ep)], 2), Lh[0]["n_train"]])
    return ["System", "Model", "Parameters", "Epochs run", "Early-stopped", "Best val loss", "Train time (s)", "s / epoch", "N train"], rows, \
        "[main] = full loading range, [extrap] = trained below the loading quantile only."

def st_latency(Rs):
    rows = []
    for R in Rs:
        rows.append(R.system)
        for m in ORDER:
            L = R.lat.get(m, [])
            if not L: continue
            rows.append([R.system, DISPLAY[m], f"{R.train_info[m][0]['params']:,}", S([x["b256"] for x in L], 3), S([x["b1"] for x in L], 3)])
        rows.append([R.system, "External Pi_phys Newton (python, per scenario)", "--", "--", _f(getattr(R, "t_newton", float("nan")), 3)])
    return ["System", "Method", "Parameters", "ms / scenario (batch 256)", "ms / scenario (batch 1, median)"], rows, \
        "Inference time of the full forward pass incl. the projection layer."

def st_percat(Rs):
    rows = []
    for R in Rs:
        rows.append(R.system)
        for m in ORDER:
            for c in range(3):
                mk = R.cats_te == c
                if not mk.any(): continue
                r = R.main[m]
                rows.append([R.system, f"{DISPLAY[m]} / {LEVEL_NAMES[c]}"] + [S(sv(r, "raw", mk, k), 1) for k in ("rmse_pg", "rmse_qg", "rmse_v", "rmse_th")] +
                            [S(sv(r, "raw", mk, "gap"), 3), S(sv(r, "raw", mk, "gap_p95"), 3), S(sv(r, "raw", mk, "feas"), 1), int(mk.sum())])
    return ["System", "Model / loading level", "RMSE pg", "RMSE qg", "RMSE v", "RMSE theta", "Gap mean(%)", "Gap p95(%)", "Feas raw(%)", "n test"], rows, \
        "RMSE in 1e-5 p.u."

def st_recovery(Rs):
    rows = []
    for R in Rs:
        rows.append(R.system); mk = _allm(R)
        for m in ORDER:
            r = R.main[m]; g = lambda v, k: _mm(r, v, mk, k)
            fr_, fc_ = g("raw", "feas"), g("rec", "feas")
            rows.append([R.system, DISPLAY[m], _f(fr_, 1), _f(fc_, 1), _f(fc_ - fr_, 1), _f(g("raw", "gap"), 3), _f(g("rec", "gap"), 3),
                         _f(g("raw", "r_p"), 2, True), _f(g("rec", "r_p"), 2, True), _f(g("raw", "max_viol"), 3), _f(g("rec", "max_viol"), 3)])
    return ["System", "Model", "Feas raw", "Feas rec", "Delta(pp)", "Gap raw", "Gap rec", "r_P raw", "r_P rec", "Maxviol raw", "Maxviol rec"], rows, \
        "External classical recovery effect."

def st_projection(Rs):
    rows = []
    for R in Rs:
        rows.append(R.system); mk = _allm(R)
        for m in NET:
            r = R.main[m]; a, p = _mm(r, "pre", mk, "r_p"), _mm(r, "raw", mk, "r_p")
            rows.append([R.system, DISPLAY[m], _f(a, 2, True), _f(p, 2, True), _f(a / p if p > 0 else float("nan"), 1), _f(_mm(r, "pre", mk, "feas"), 1),
                         _f(_mm(r, "raw", mk, "feas"), 1), _f(_mm(r, "pre", mk, "gap"), 3), _f(_mm(r, "raw", mk, "gap"), 3)])
        m = "Unsup-FNO-Soft"; r = R.main[m]
        rows.append([R.system, DISPLAY[m] + " (no layer)", _f(_mm(r, "raw", mk, "r_p"), 2, True), "--", "--", _f(_mm(r, "raw", mk, "feas"), 1), "--",
                     _f(_mm(r, "raw", mk, "gap"), 3), "--"])
    return ["System", "Model", "r_P before layer", "r_P after layer", "Reduction (x)", "Feas before", "Feas after", "Gap before", "Gap after"], rows, \
        "Effect of the built-in projection layer."

def st_sweep(Rs):
    rows = []
    for R in Rs:
        if not R.sweep: continue
        rows.append(R.system); mk = _allm(R)
        for K in SWEEP_K:
            runs = [dict(raw=M) for M in R.sweep.get(K, [])]
            if not runs: continue
            rows.append([R.system, f"K = {K}" + (" (bounds only)" if K == 0 else "") + (" (trained)" if K == PROJ_IT else ""),
                         S(sv(runs, "raw", mk, "r_p"), 2, True), S(sv(runs, "raw", mk, "max_eq"), 1, True), S(sv(runs, "raw", mk, "feas"), 1),
                         S(sv(runs, "raw", mk, "gap"), 3), S(sv(runs, "raw", mk, "csr_gen"), 1), S(sv(runs, "raw", mk, "csr_volt"), 1)])
    return ["System", "PNO-FNO inference iterations", "r_P", "Max eq.resid", "Feas(%)", "Gap(%)", "GenBox(%)", "Voltage(%)"], rows, \
        "The trained PNO-FNO is evaluated with a different number of projection iterations at inference."

def st_signif(Rs):
    pgk, vk, gk, rk = (lambda M: M["se_pg"]), (lambda M: np.maximum(M["r_bal"], _arr(M, "r_in"))), (lambda M: M["gap"]), (lambda M: M["r_p"])
    rows = []
    for R in Rs:
        rows.append(R.system); allm = _allm(R)
        metrics = [("Cost gap (%) raw", "gap", "raw", True, gk, 3), ("r_P raw", "r_p", "raw", True, rk, 3),
                   ("Strict feas raw (%)", "feas", "raw", False, vk, 1), ("RMSE pg raw", "rmse_pg", "raw", True, pgk, 1)]
        for base in [m for m in ORDER if m != "PNO-FNO"]:
            for lab, key, var, low, fn, p in metrics:
                a, b_ = sv(R.main["PNO-FNO"], var, allm, key), sv(R.main[base], var, allm, key)
                if not a or not b_: continue
                better = sum((x < y) if low else (x > y) for x, y in zip(a, b_))
                wr = _winrate(R.main["PNO-FNO"], R.main[base], var, allm, fn)
                rows.append([R.system, f"PNO-FNO vs {DISPLAY[base]}", lab, _f(_mean(a), p, key == "r_p"), _f(_mean(b_), p, key == "r_p"),
                             _f(_mean(a) - _mean(b_), p, key == "r_p"), _f(_pval(a, b_), 4), f"{better}/{len(a)}", _f(wr, 1)])
    return ["System", "Comparison", "Metric", "PNO-FNO", "Other", "Delta", "p (paired t, runs)", "PNO-FNO better in runs", "Per-scenario win-rate (%)"], rows, \
        "p-values need >=2 runs."

def st_claims(Rs):
    oth = [m for m in NET if m != "PNO-FNO"]
    mm = lambda R, m, v, k, mk=None: _mm(R.main[m], v, _allm(R) if mk is None else mk, k)
    lat = lambda R, m: _mean([x["b256"] for x in R.lat[m]])
    Oall = lambda R, f: all(f(R, o) for o in oth)
    C = [
        ("Soft-penalty unsupervised FNO has ~0 % strict feasibility (< 1 %)", lambda R: mm(R, "Unsup-FNO-Soft", "raw", "feas") < 1.0),
        ("PNO lifts strict feasibility of ALL four backbones above 98 %", lambda R: all(mm(R, m, "raw", "feas") > 98.0 for m in NET)),
        ("Box constraints hold by construction for all PNO models", lambda R: all(mm(R, m, "raw", "csr_gen") >= 99.99 and mm(R, m, "raw", "csr_volt") >= 99.99 for m in NET)),
        ("PNO-FNO has the lowest cost gap among the PNO backbones", lambda R: Oall(R, lambda R_, o: mm(R_, "PNO-FNO", "raw", "gap") < mm(R_, o, "raw", "gap"))),
        ("PNO-FNO has the highest strict feasibility among the PNO backbones", lambda R: Oall(R, lambda R_, o: mm(R_, "PNO-FNO", "raw", "feas") >= mm(R_, o, "raw", "feas"))),
        ("PNO-FNO has the lowest latency among the PNO backbones", lambda R: Oall(R, lambda R_, o: lat(R_, "PNO-FNO") < lat(R_, o))),
        ("PNO-DeepONet is slower than PNO-FNO", lambda R: lat(R, "PNO-DeepONet") > lat(R, "PNO-FNO")),
        ("PNO-WNO is slower than PNO-FNO", lambda R: lat(R, "PNO-WNO") > lat(R, "PNO-FNO")),
        ("PNO-GNO has the largest cost gap among the PNO backbones", lambda R: all(mm(R, "PNO-GNO", "raw", "gap") > mm(R, o, "raw", "gap") for o in NET if o != "PNO-GNO")),
        ("All PNO models have mean cost gap < 1 %", lambda R: all(mm(R, m, "raw", "gap") < 1.0 for m in NET)),
        ("Soft-penalty FNO has the largest cost gap of all learned models", lambda R: all(mm(R, "Unsup-FNO-Soft", "raw", "gap") > mm(R, m, "raw", "gap") for m in ORDER if m != "Unsup-FNO-Soft")),
        ("PNO-FNO has higher strict feasibility than the supervised MLP", lambda R: mm(R, "PNO-FNO", "raw", "feas") > mm(R, "Sup-MLP", "raw", "feas")),
        ("The supervised MLP has a lower cost gap than every PNO model", lambda R: all(mm(R, "Sup-MLP", "raw", "gap") < mm(R, m, "raw", "gap") for m in NET)),
        ("The projection layer reduces the P-balance residual of PNO-FNO", lambda R: mm(R, "PNO-FNO", "raw", "r_p") < mm(R, "PNO-FNO", "pre", "r_p")),
        ("PNO-FNO lands close to the manifold before projection (r_P < 1e-2 p.u.)", lambda R: mm(R, "PNO-FNO", "pre", "r_p") < 1e-2),
        ("PNO-FNO final P-balance residual r_P < 1e-3 p.u.", lambda R: mm(R, "PNO-FNO", "raw", "r_p") < 1e-3),
        ("PNO-FNO latency is below 1 ms per scenario", lambda R: lat(R, "PNO-FNO") < 1.0),
        ("Removing the projection layer lowers strict feasibility of PNO-FNO", lambda R: None if not R.abl else _mm(R.abl["w/o projection layer"], "raw", _allm(R), "feas") < _mm(R.abl["Full PNO-FNO"], "raw", _allm(R), "feas")),
        ("Removing the sigmoid bounds lowers box satisfaction before projection", lambda R: None if not R.abl else min(_mm(R.abl["w/o sigmoid bounds"], "pre", _allm(R), "csr_gen"), _mm(R.abl["w/o sigmoid bounds"], "pre", _allm(R), "csr_volt")) < min(_mm(R.abl["Full PNO-FNO"], "pre", _allm(R), "csr_gen"), _mm(R.abl["Full PNO-FNO"], "pre", _allm(R), "csr_volt"))),
        ("Training through the projection beats attaching it only at inference", lambda R: None if not R.abl else _mm(R.abl["Full PNO-FNO"], "raw", _allm(R), "gap") < _mm(R.abl["projection at inference only"], "raw", _allm(R), "gap")),
        ("PNO-FNO keeps strict feasibility > 95 % under load extrapolation", lambda R: None if not R.extrap["PNO-FNO"] else _mm(R.extrap["PNO-FNO"], "raw", np.ones(len(R.extrap["PNO-FNO"][0]["raw"]["gap"]), bool), "feas") > 95.0),
    ]
    rows = []
    for i, (txt, fn) in enumerate(C, 1):
        res = []
        for R in Rs:
            try: x = fn(R)
            except Exception: x = None
            res.append(x)
        ok = sum(1 for x in res if x is True); n = sum(1 for x in res if x is not None)
        rows.append([i, txt] + [("n/a" if x is None else ("PASS" if x else "FAIL")) for x in res] + [f"{ok}/{n}"])
    tot = sum(1 for r in rows for c in r[2:-1] if c == "PASS"); den = sum(1 for r in rows for c in r[2:-1] if c != "n/a")
    rows.append(["", "TOTAL"] + [""] * len(Rs) + [f"{tot}/{den}"])
    return ["#", "Claim made in the article (checked on this run)"] + [R.system for R in Rs] + ["Systems passing"], rows, \
        "Automatic check of the article's qualitative claims against this run."

SUPP_TABLES = [
    ("environment", "Reproducibility: environment and settings", st_env),
    ("data_integrity", "Dataset structure and integrity checks (leakage, duplicates, labels)", st_integrity),
    ("training_behaviour", "Model size and training behaviour (epochs, early stopping, time)", st_training),
    ("latency", "Inference latency (batched and single scenario) and parameter counts", st_latency),
    ("per_loading_level", "Accuracy in all three loading levels", st_percat),
    ("recovery_effect", "Effect of an external Pi_phys recovery on every model", st_recovery),
    ("projection_effect", "Effect of the built-in projection layer (before vs after)", st_projection),
    ("projection_sweep", "Projection-iteration sweep at inference (PNO-FNO)", st_sweep),
    ("significance", "Statistical comparison of PNO-FNO with every other model", st_signif),
    ("claim_check", "Automatic verification of the article's qualitative claims", st_claims),
]

def emit_header(Rs):
    emit("=" * 100)
    emit("UNIFIED PROJECTED NEURAL OPERATOR (PNO) AC-OPF STUDY - RESULT TABLES ON FRESHLY-SOLVED RANDOMIZED AC-OPF SCENARIOS")
    emit("=" * 100)
    emit(f"generated : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    emit(f"systems   : {', '.join(R.system for R in Rs)}  |  N_DATA={N_DATA}  seeds={N_SEEDS}  max epochs={EPOCHS}  device={DEVICE}")
    emit(f"models    : {', '.join(ORDER)}")
    emit("NOTE      : reference labels are generated by running a fresh nonlinear AC-OPF for randomized load scenarios; no pre-solved OPF dataset is used.")
    emit("           PYPOWER AC-OPF solutions provide reference labels; surrogate predictions, metrics, and tables are computed by this script.")

def print_supplementary(Rs):
    emit("\n" + "=" * 100); emit("SUPPLEMENTARY TABLES"); emit("=" * 100); emit("Contents:")
    for i, (_, title, _) in enumerate(SUPP_TABLES, 1): emit(f"  S{i:<2} {title}")
    for i, (slug, title, fn) in enumerate(SUPP_TABLES, 1):
        head = f"Supplementary Table S{i}. {title}"
        try: header, rows, note = fn(Rs)
        except Exception as e:
            emit(f"\n{head}\n  [skipped: {type(e).__name__}: {e}]"); continue
        if not header or not any(not isinstance(r, str) for r in rows):
            emit(f"\n{head}\n  [no data available - needs a fresh run (not --mode tables on old cached results)]"); continue
        try: table(f"tableS{i}_{slug}", head, header, rows, note)
        except Exception as e: emit(f"\n{head}\n  [could not be formatted: {type(e).__name__}: {e}]")

# =====================================================================================
#  MAIN
# =====================================================================================
def main():
    global SYSTEMS, N_SAMPLES, N_SEEDS, EPOCHS, PATIENCE, ABL_SYSTEM, ABL_SEEDS, EXTRAP_SEEDS, N_DATA
    ap = argparse.ArgumentParser(description="Settings live at the top of the file; flags only override them.")
    ap.add_argument("--mode", choices=["data", "all", "tables"], default=None)
    ap.add_argument("--cases", "--systems", nargs="+", dest="cases", default=None, choices=list(CASES))
    ap.add_argument("--list-cases", action="store_true")
    ap.add_argument("--n-samples", type=int, default=None); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force-regen", action="store_true")
    ap.add_argument("--seeds", type=int, default=None); ap.add_argument("--abl-seeds", type=int, default=None); ap.add_argument("--extrap-seeds", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None); ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--abl-system", default=None)
    a, _ = ap.parse_known_args()
    if a.list_cases:
        print("Available PYPOWER cases (set CASE_SELECT at the top, or use --cases ...):")
        for k, v in CASES.items(): print(f"  {k:8s} -> {v}")
        return
    mode = a.mode or RUN_MODE; force = a.force_regen or FORCE_REGEN
    bad = [c for c in CASE_SELECT if c not in CASES]
    if bad: raise SystemExit(f"Unknown case(s) in CASE_SELECT: {bad}. Valid: {list(CASES)}")
    SYSTEMS = list(a.cases) if a.cases else list(CASE_SELECT)
    if a.seeds is not None: N_SEEDS = a.seeds
    if a.abl_seeds is not None: ABL_SEEDS = a.abl_seeds
    if a.extrap_seeds is not None: EXTRAP_SEEDS = a.extrap_seeds
    if a.epochs is not None: EPOCHS = a.epochs
    if a.patience is not None: PATIENCE = a.patience
    if a.abl_system: ABL_SYSTEM = a.abl_system
    if ABL_SYSTEM not in SYSTEMS: ABL_SYSTEM = SYSTEMS[-1]
    N_SAMPLES = {s: (a.n_samples or N_DATA) for s in SYSTEMS}
    if a.quick or QUICK_TEST: N_SAMPLES = {s: 300 for s in SYSTEMS}; N_SEEDS, EPOCHS, PATIENCE = 1, 12, 6
    torch.manual_seed(SEED); np.random.seed(SEED)
    print(f"mode={mode} systems={SYSTEMS} device={DEVICE} samples={ {s: target_n(s) for s in SYSTEMS} } seeds={N_SEEDS} epochs={EPOCHS} models={ORDER}")
    os.makedirs(RES_DIR, exist_ok=True); Rs = []; ctxs = {}
    if mode != "tables":
        for s in SYSTEMS:
            print(f"\n=== DATA ({s}) ==="); do_data(s, force); ctxs[s] = load_system(s)
    for s in SYSTEMS:
        pk = f"{RES_DIR}/out_{s}.pkl"
        if mode == "tables":
            if os.path.exists(pk):
                with open(pk, "rb") as f: Rs.append(pickle.load(f))
            else: print(f"  no cached results for {s}")
            continue
        if mode == "data": continue
        ctx = ctxs[s]
        print(f"=== TRAIN / EVAL ({s}) : {len(ctx.A)} samples, {ctx.b['n']} buses ===")
        t0 = time.time(); R = run_system(ctx, N_SEEDS, ABL_SEEDS); print(f"  system time: {time.time()-t0:.0f}s")
        with open(pk, "wb") as f: pickle.dump(R, f)
        Rs.append(R)
    if Rs: print("\n" + "=" * 100 + "\nALL TABLES\n" + "=" * 100); print_all(Rs)

if __name__ == "__main__":
    main()
