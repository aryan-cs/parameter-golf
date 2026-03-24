from __future__ import annotations
import copy, glob, io, json, lzma, math, os, random, struct, subprocess, sys, time, uuid, zlib
try:
    import zstandard as zstd; HAS_ZSTD = True
except ImportError: HAS_ZSTD = False
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch as th, torch.distributed as dist, torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
EG=os.environ.get; JP=os.path.join; PC=time.perf_counter
BF=th.bfloat16; F16=th.float16; F32=th.float32; F64=th.float64; I8=th.int8; I16=th.int16; I64=th.int64; BO=th.bool; U16=th.uint16
AC=th.autocast; IM=th.inference_mode; Z=th.zeros; TT=th.tensor; QT=th.quantile; EM=th.empty; FN=th.from_numpy; SK=th.stack; PK=struct.pack; UF=struct.unpack_from; ER=ValueError; CU="cuda"; AU=lambda e=1:AC(device_type=CU,dtype=BF,enabled=e); DC=lambda t:t.detach().cpu(); DL=lambda t:DC(t).clone(); DX=lambda t:DC(t).contiguous(); AS=lambda x,d:x.astype(d,copy=False)
P=nn.Parameter; PL=nn.ParameterList; M=nn.Module; ML=nn.ModuleList; NI=nn.init; NG=th.no_grad; CLP=th.clamp; DG=th.diag; CMP=th.compile; SY=th.cuda.synchronize
IA=dist.is_available; II=dist.is_initialized; BR=dist.barrier; GWS=dist.get_world_size; GRK=dist.get_rank; IGP=dist.init_process_group; DGP=dist.destroy_process_group; ARD=dist.all_reduce; ROP=dist.ReduceOp
ZL=th.zeros_like; EL=th.empty_like; AR=th.arange; SG=th.sigmoid; CAT=th.cat; ON=th.ones; FUL=th.full; EYE=th.eye
N8=np.uint8; N6=np.int16; N1=np.int8
try:
    from flash_attn_interface import flash_attn_func as _fa3_func
    HAS_FA3 = True
except ImportError:
    HAS_FA3 = False

class H:
    dp = EG("DP", "./data/datasets/fineweb10B_sp1024")
    tf = JP(dp, "fineweb_train_*.bin")
    vf = JP(dp, "fineweb_val_*.bin")
    tp = EG("TP", "./data/tokenizers/fineweb_1024_bpe.model")
    rid = EG("RUN_ID", str(uuid.uuid4()))
    sd = int(EG("SEED", 42))
    vbs = int(EG("VBS", 524_288))
    vle = int(EG("VLE", 500))
    tle = int(EG("TLE", 200))
    it = int(EG("IT", 20000))
    wus = int(EG("WUS", 20))
    tbt = int(EG("TBT", 786_432))
    tsl = int(EG("TSL", 2048))
    esl = int(EG("ESL", 2048))
    evs = int(EG("EVS", 64))
    ebs = int(EG("EBS", 256))
    mws = float(EG("MWS", 600.0))
    qgi = float(EG("QGI", 1.5))
    vs = int(EG("VS", 1024))
    nl = int(EG("NL", 11))
    nkh = int(EG("NKH", 4))
    dm = int(EG("DM", 512))
    nh = int(EG("NH", 8))
    mm = int(EG("MM", 3))
    te = bool(int(EG("TE", "1")))
    rb = float(EG("RB", 10000.0))
    lsc = float(EG("LSC", 30.0))
    elr = float(EG("ELR", 0.6))
    hlr = float(EG("HLR", 0.008))
    telr = float(EG("TELR", 0.035))
    teis = float(EG("TEIS", 0.005))
    mlr = float(EG("MLR", 0.025))
    slr = float(EG("SLR", 0.025))
    mum = float(EG("MUM", 0.99))
    mns = int(EG("MNS", 5))
    mmst = float(EG("MMST", 0.92))
    mmws = int(EG("MMWS", 1500))
    mwd = float(EG("MWD", 0.04))
    beta1 = float(EG("B1", 0.9))
    beta2 = float(EG("B2", 0.95))
    aep = float(EG("AEP", 1e-8))
    awd = float(EG("AWD", 0.04))
    gcn = float(EG("GCN", 0.3))
    se = bool(int(EG("SE", "1")))
    be = bool(int(EG("BE", "0")))
    boi = float(EG("BOI", 0.2))
    ed = float(EG("ED", 0.997))
    swe = bool(int(EG("SWE", "1")))
    swi = int(EG("SWI", 50))
    wdi = int(EG("WDI", 3500))
    lqt = float(EG("LQT", 0.15))
    bgvs = int(EG("BGVS", 2048))
    bgd = int(EG("BGD", 128))
    xsn = int(EG("XSN", 11))
    rd = int(EG("RD", 16))
    lns = bool(int(EG("LNS", "1")))
    vee = bool(int(EG("VEE", "1")))
    ved = int(EG("VED", 128))
    vel = EG("VEL", "9,10")
    gcb = int(EG("GCB", 256))
    gbs = int(EG("GBS", 128))
    qcp = float(EG("QCP", 0.9995))
    pp = float(EG("PP", 0.02))
    spc = EG("SPC", "")
    eoc = EG("EOC", "")

def z5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16(); X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed: X = X.T
    for _ in range(steps):
        A = X @ X.T; B = b * A + c * A @ A; X = a * X + B @ X
    return X.T if transposed else X

class MU(th.optim.Optimizer):
    def __init__(self, params, lr, momentum, ns_steps, wd=0.0, nesterov=True):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps, wd=wd, nesterov=nesterov))
    @NG()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with th.enable_grad(): loss = closure()
        dd = IA() and II()
        ws = GWS() if dd else 1
        rk = GRK() if dd else 0
        for group in self.param_groups:
            params = group["params"]
            if not params: continue
            lr, momentum, ns_steps = group["lr"], group["momentum"], group["ns_steps"]
            nesterov = group["nesterov"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = Z(total_params, device=params[0].device, dtype=BF)
            curr = 0
            for i, p in enumerate(params):
                if i % ws == rk and p.grad is not None:
                    g = p.grad; state = self.state[p]
                    if "momentum_buffer" not in state: state["momentum_buffer"] = ZL(g)
                    buf = state["momentum_buffer"]; buf.mul_(momentum).add_(g)
                    if nesterov: g = g.add(buf, alpha=momentum)
                    g = z5(g, steps=ns_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if dd: ARD(updates_flat, op=ROP.SUM)
            wd = group.get("wd", 0.0); curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                if wd > 0: p.data.mul_(1.0 - lr * wd)
                p.add_(g, alpha=-lr); curr += p.numel()
        return loss

def bsl(sp, vs, dv):
    sv = int(sp.vocab_size()); ts = max(sv, vs)
    bbn = np.zeros((ts,), dtype=N6)
    hln = np.zeros((ts,), dtype=np.bool_)
    ibn = np.ones((ts,), dtype=np.bool_)
    for tid in range(sv):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid): continue
        ibn[tid] = False
        if sp.is_byte(tid): bbn[tid] = 1; continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"): hln[tid] = True; piece = piece[1:]
        bbn[tid] = len(piece.encode("utf-8"))
    return (TT(bbn, dtype=I16, device=dv),
            TT(hln, dtype=BO, device=dv),
            TT(ibn, dtype=BO, device=dv))

def lvt(pattern, sl):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files: raise FileNotFoundError(f"no:{pattern}")
    tokens = CAT([lds(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // sl) * sl
    if usable <= 0: raise ER(f"val<{sl}")
    return tokens[:usable + 1]

def eval_val(a, model, rank, ws, dv, gas,
             vt, bb, hs, ib, esl=0):
    sl = esl if esl > 0 else a.tsl
    lbs = a.vbs // (ws * gas) // sl
    tsq = (vt.numel() - 1) // sl
    ss = (tsq * rank) // ws; se = (tsq * (rank + 1)) // ws
    vls = Z((), device=dv, dtype=F64)
    vtc = Z((), device=dv, dtype=F64)
    vbc = Z((), device=dv, dtype=F64)
    model.eval()
    with IM():
        for bss in range(ss, se, lbs):
            bse = min(bss + lbs, se)
            local = vt[bss*sl:(bse*sl)+1].to(device=dv, dtype=I64, non_blocking=True)
            x, y = local[:-1].reshape(-1, sl), local[1:].reshape(-1, sl)
            with AU():
                bl = model(x, y).detach()
            vls += bl.to(F64) * float(y.numel())
            vtc += float(y.numel())
            tb = bb[y.reshape(-1)].to(dtype=I16)
            tb += (hs[y.reshape(-1)] & ~ib[x.reshape(-1)]).to(dtype=I16)
            vbc += tb.to(F64).sum()
    if IA() and II():
        for t in [vls, vtc, vbc]: ARD(t, op=ROP.SUM)
    val_loss = vls / vtc
    bpt = val_loss.item() / math.log(2.0); tpb = vtc.item() / vbc.item()
    model.train(); return float(val_loss.item()), float(bpt * tpb)

CP = tuple(
    p for p in "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear,backout_lambda,bigram.scale,ve_layer_scales,ve_shared.scale,vrl_alphas".split(",") if p)
I8K = 65_536
I8D = F16
MP = "p"
MC = "c"
M6 = 6
M8 = 8
QMB = b"QMB1"
QCB = b"QCB1"
KZ = 1
KG = 2
KL = 3
LF = [{
    "id": lzma.FILTER_LZMA2,
    "dict_size": 1 << 24,
    "lc": 3,
    "lp": 0,
    "pb": 2,
    "mode": lzma.MODE_NORMAL,
    "nice_len": 273,
    "mf": lzma.MF_HC3,
    "depth": 0,
}]

def cpm(name):
    if "tok_emb" in name or "lm_head" in name: return "embed"
    if ".mlp." in name: return "mlp"
    if "bigram" in name: return "bigram"
    if ".attn." in name or (".proj." in name and ".mlp." not in name): return "attn"
    if "ve_shared" in name: return "ve"
    return "other"

def mk(info):
    if info == MP or info == "passthrough":
        return "passthrough"
    if info == MC or info == "passthrough_ctrl":
        return "passthrough_ctrl"
    if info == M6 or (isinstance(info, dict) and info.get("type") == "int6"):
        return "int6"
    if info == M8 or (isinstance(info, dict) and info.get("type") == "int8"):
        return "int8"
    return None

def eqm(meta: dict[str, object]) -> tuple[bytes, list[str]]:
    names = sorted(meta)
    kind_map = {
        MP: 0,
        MC: 1,
        M6: 2,
        M8: 3,
        "passthrough": 0,
        "passthrough_ctrl": 1,
    }
    parts = [QMB, PK("<H", len(names))]
    for name in names:
        kind = meta[name]
        kind_code = kind_map.get(kind)
        if kind_code is None:
            kind_name = mk(kind)
            if kind_name == "int6": kind_code = 2
            elif kind_name == "int8": kind_code = 3
            else: raise ER(f"bad meta {name}:{kind!r}")
        name_bytes = name.encode("utf-8")
        parts.append(PK("<HB", len(name_bytes), kind_code))
        parts.append(name_bytes)
    return b"".join(parts), names

def dqm(blob: bytes) -> tuple[dict[str, object], list[str], bool]:
    if blob.startswith(QMB):
        o = len(QMB)
        entry_count = UF("<H", blob, o)[0]; o += 2
        kind_map = {
            0: MP,
            1: MC,
            2: M6,
            3: M8,
        }
        meta, names = {}, []
        for _ in range(entry_count):
            name_len, kind_code = UF("<HB", blob, o); o += 3
            name = blob[o:o+name_len].decode("utf-8"); o += name_len
            meta[name] = kind_map[kind_code]
            names.append(name)
        return meta, names, True
    return json.loads(blob.decode("utf-8")), [], False

def etr(tname: str, nti: dict[str, int]) -> tuple[int, int]:
    if tname in nti:
        return nti[tname], 0
    if tname.endswith(".q") and tname[:-2] in nti:
        bn, suffix = tname[:-2], 1
    elif tname.endswith(".scale") and tname[:-6] in nti:
        bn, suffix = tname[:-6], 2
    else:
        bn, suffix = tname, 0
    return nti[bn], suffix

def dtr(name_idx: int, suffix: int, names: list[str]) -> str:
    bn = names[name_idx]
    if suffix == 1: return bn + ".q"
    if suffix == 2: return bn + ".scale"
    return bn

def qi6g(weight, hessian=None, cr=31, block_size=128):
    t32 = weight.float()
    if t32.ndim != 2 or hessian is None:
        return qi6p(t32, cr)
    rows, cols = t32.shape
    H = hessian.float().clone()
    dead = DG(H) == 0
    H[dead, dead] = 1
    damp = 0.01 * th.mean(DG(H))
    H[AR(cols), AR(cols)] += damp
    perm = th.argsort(DG(H), descending=True)
    inv_perm = th.argsort(perm)
    W = t32[:, perm].clone()
    W[:, dead[perm]] = 0
    H = H[perm][:, perm]
    try:
        Hinv = th.linalg.cholesky(H)
        Hinv = th.cholesky_inverse(Hinv)
        Hinv = th.linalg.cholesky(Hinv, upper=True)
    except th.linalg.LinAlgError:
        return qi6p(t32, cr)
    best_q = None; best_scale = None; best_err = float('inf')
    for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
        if pct < 1.0:
            row_clip = QT(t32.abs(), pct, dim=1)
        else:
            row_clip = t32.abs().amax(dim=1)
        s = (row_clip / cr).clamp_min(1.0 / cr).to(F16)
        sf = s.float()
        Q = ZL(W, dtype=I8)
        W_work = W.clone()
        for i1 in range(0, cols, block_size):
            i2 = min(i1 + block_size, cols)
            count = i2 - i1
            W1 = W_work[:, i1:i2].clone()
            Q1 = Z(rows, count, dtype=I8)
            Err1 = Z(rows, count)
            Hinv1 = Hinv[i1:i2, i1:i2]
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                q = CLP(th.round(w / sf), -cr, cr).to(I8)
                Q1[:, i] = q
                err = (w - q.float() * sf) / d
                W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
                Err1[:, i] = err
            Q[:, i1:i2] = Q1
            if i2 < cols:
                W_work[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
        recon = Q.float() * sf[:, None]
        mse = (W - recon).pow(2).mean().item()
        if mse < best_err:
            best_q, best_scale, best_err = Q, s, mse
    best_q = best_q[:, inv_perm]
    return best_q, best_scale

def qi6p(t32, cr=31):
    if t32.ndim == 2:
        best_q, best_s, best_err = None, None, float('inf')
        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
            if pct < 1.0:
                row_clip = QT(t32.abs(), pct, dim=1)
            else:
                row_clip = t32.abs().amax(dim=1)
            s = (row_clip / cr).clamp_min(1.0 / cr).to(F16)
            q = CLP(th.round(t32 / s.float()[:, None]), -cr, cr).to(I8)
            recon = q.float() * s.float()[:, None]
            err = (t32 - recon).pow(2).mean().item()
            if err < best_err:
                best_q, best_s, best_err = q, s, err
        return best_q, best_s
    amax = t32.abs().max().item()
    scale = TT(amax / cr if amax > 0 else 1.0, dtype=F16)
    q = CLP(th.round(t32 / scale.float()), -cr, cr).to(I8)
    return q, scale

def qft(t):
    t32 = t.float()
    if t32.ndim == 2:
        clip_q = 99.99984 / 100.0
        clip_abs = QT(t32.abs(), clip_q, dim=1) if t32.numel() else EM((t32.shape[0],), dtype=F32)
        clipped = th.maximum(th.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = CLP(th.round(clipped / scale[:, None]), -127, 127).to(I8).contiguous()
        return q, scale.to(dtype=I8D).contiguous()
    clip_q = 99.99984 / 100.0
    clip_abs = float(QT(t32.abs().flatten(), clip_q).item()) if t32.numel() else 0.0
    scale = TT(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=F32)
    q = CLP(th.round(CLP(t32, -clip_abs, clip_abs) / scale), -127, 127).to(I8).contiguous()
    return q, scale

def qsd(sd, hh=None):
    result, meta = {}, {}
    int6_cats = {"mlp", "attn", "bigram", "ve"}
    for name, tensor in sd.items():
        t = DX(tensor)
        cat = cpm(name)
        if not t.is_floating_point() or t.numel() <= I8K:
            result[name] = t.to(F16) if t.is_floating_point() else t
            meta[name] = MP; continue
        if any(p in name for p in CP):
            result[name] = t.float(); meta[name] = MC; continue
        if cat in int6_cats and t.ndim >= 1:
            H = hh.get(name) if hh else None
            q, s = qi6g(t, hessian=H)
            result[name + ".q"] = q; result[name + ".scale"] = s
            meta[name] = M6; continue
        q, s = qft(t)
        result[name + ".q"] = q; result[name + ".scale"] = s
        meta[name] = M8
    return result, meta

def dsd(result, meta, template_sd):
    out = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None: continue
        orig_dtype = orig.dtype
        if mk(info) in {"passthrough", "passthrough_ctrl"}:
            t = result[name]
            if t.dtype == F16 and orig_dtype in (F32, BF): t = t.to(orig_dtype)
            out[name] = t; continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1]*(q.ndim-1)))).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
    return out

def _zigzag_encode_int6(arr_i16: np.ndarray) -> np.ndarray:
    arr = AS(arr_i16, N6)
    if arr.size == 0:
        return np.empty((0,), dtype=N8)
    if arr.min() < -31 or arr.max() > 31:
        raise ER("int6 range")
    out = np.where(arr >= 0, arr * 2, (-arr) * 2 - 1)
    return AS(out, N8)

def _zigzag_decode_int6(u8: np.ndarray) -> np.ndarray:
    u = AS(u8, N6)
    return AS(np.where((u & 1) == 0, u // 2, -((u + 1) // 2)), N6)

def pi6l(t: Tensor) -> bytes:
    q = DX(t)
    if q.dtype != I8:
        raise TypeError(f"need int8, got {q.dtype}")
    arr = AS(q.numpy().reshape(-1), N6)
    if arr.size == 0:
        return b""
    if arr.min() < -31 or arr.max() > 31:
        raise ER("int6 range")
    u = AS(arr + 31, N8)
    pad = (-u.size) % 4
    if pad:
        u = np.pad(u, (0, pad), constant_values=0)
    groups = AS(u.reshape(-1, 4), np.uint32)
    packed = (groups[:, 0]
              | (groups[:, 1] << 6)
              | (groups[:, 2] << 12)
              | (groups[:, 3] << 18))
    out = np.empty(packed.size * 3, dtype=N8)
    out[0::3] = (packed & 0xFF).astype(N8)
    out[1::3] = ((packed >> 8) & 0xFF).astype(N8)
    out[2::3] = ((packed >> 16) & 0xFF).astype(N8)
    return out.tobytes()

def ui6l(raw: bytes | memoryview, shape: list[int]) -> Tensor:
    numel = int(np.prod(shape, dtype=np.int64))
    if numel == 0:
        return EM(shape, dtype=I8)
    packed_u8 = np.frombuffer(raw, dtype=N8)
    groups = (numel + 3) // 4
    if packed_u8.size != groups * 3:
        raise ER(f"bad int6 sz {groups*3}!={packed_u8.size}")
    triplets = AS(packed_u8.reshape(-1, 3), np.uint32)
    packed = triplets[:, 0] | (triplets[:, 1] << 8) | (triplets[:, 2] << 16)
    u = np.empty(groups * 4, dtype=N8)
    u[0::4] = (packed & 0x3F).astype(N8)
    u[1::4] = ((packed >> 6) & 0x3F).astype(N8)
    u[2::4] = ((packed >> 12) & 0x3F).astype(N8)
    u[3::4] = ((packed >> 18) & 0x3F).astype(N8)
    arr = AS(u[:numel], N6) - 31
    return FN(AS(arr, N1).reshape(shape))

def pi6(t: Tensor) -> bytes:
    q = DX(t)
    if q.dtype != I8:
        raise TypeError(f"need int8, got {q.dtype}")
    arr = AS(q.numpy().reshape(-1), N6)
    if arr.size == 0:
        return b""
    u = _zigzag_encode_int6(AS(arr, N6))
    numel = u.size
    pad = (-numel) % 8
    if pad:
        u = np.pad(u, (0, pad), constant_values=0)
    out = bytearray()
    for bit in range(6):
        bits = AS((u >> bit) & 1, N8).reshape(-1, 8)
        out.extend(np.packbits(bits, axis=1, bitorder="little").reshape(-1).tobytes())
    return bytes(out)

def ui6(raw: bytes | memoryview, shape: list[int]) -> Tensor:
    numel = int(np.prod(shape, dtype=np.int64))
    if numel == 0:
        return EM(shape, dtype=I8)
    padded = ((numel + 7) // 8) * 8
    plane_bytes = padded // 8
    packed_u8 = np.frombuffer(raw, dtype=N8)
    expected = plane_bytes * 6
    if packed_u8.size != expected:
        raise ER(f"bad int6 sz {expected}!={packed_u8.size}")
    u = np.zeros(padded, dtype=N8)
    o = 0
    for bit in range(6):
        plane = packed_u8[o:o + plane_bytes]
        o += plane_bytes
        bits = np.unpackbits(plane, bitorder="little")
        u |= (AS(bits, N8) << bit)
    arr = _zigzag_decode_int6(u[:numel])
    return FN(AS(arr, N1).reshape(shape))

def mcn(cid: int) -> str:
    if cid == KZ:
        return "zstd19"
    if cid == KG:
        return "zlib9"
    if cid == KL:
        return "lzma_raw_hc3_16mb"
    return f"unknown({cid})"

def cmb(raw: bytes) -> tuple[bytes, int]:
    candidates = [
        (KL, lzma.compress(raw, format=lzma.FORMAT_RAW, filters=LF)),
        (KG, zlib.compress(raw, level=9)),
    ]
    if HAS_ZSTD:
        candidates.append((KZ, zstd.ZstdCompressor(level=19).compress(raw)))
    cid, payload = min(candidates, key=lambda item: len(item[1]))
    return QCB + bytes([cid]) + payload, cid

def dmb(blob: bytes) -> tuple[bytes, str]:
    if blob.startswith(QCB):
        cid = blob[len(QCB)]
        payload = blob[len(QCB) + 1:]
        if cid == KZ:
            if not HAS_ZSTD:
                raise RuntimeError("zstd unavailable")
            return zstd.ZstdDecompressor().decompress(payload), mcn(cid)
        if cid == KG:
            return zlib.decompress(payload), mcn(cid)
        if cid == KL:
            try:
                return lzma.decompress(payload, format=lzma.FORMAT_RAW, filters=LF), mcn(cid)
            except lzma.LZMAError:
                return lzma.decompress(payload), "legacy_lzma_hc4_32mb_xz"
        raise ER(f"bad codec {cid}")
    if HAS_ZSTD:
        try:
            return zstd.ZstdDecompressor().decompress(blob), "legacy_zstd22"
        except Exception:
            pass
    return zlib.decompress(blob), "legacy_zlib9"

def lds(file):
    hb = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1: raise ER(f"bad hdr {file}")
    nt = int(header[2])
    if file.stat().st_size != hb + nt * np.dtype("<u2").itemsize: raise ER(f"bad size {file}")
    return FN(AS(np.fromfile(file, dtype="<u2", count=nt, offset=hb), np.uint16))

class TS:
    def __init__(self, pat):
        self.files = [Path(p) for p in sorted(glob.glob(pat))]
        if not self.files: raise FileNotFoundError(f"no:{pat}")
        self.file_idx = 0; self.tokens = lds(self.files[0]); self.pos = 0
    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files); self.tokens = lds(self.files[self.file_idx]); self.pos = 0
    def take(self, n):
        chunks, remaining = [], n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0: self._advance_file(); continue
            k = min(remaining, avail); chunks.append(self.tokens[self.pos:self.pos+k]); self.pos += k; remaining -= k
        return chunks[0] if len(chunks) == 1 else CAT(chunks)

class DTL:
    def __init__(self, pat, rank, ws, dv):
        self.rank, self.ws, self.dv = rank, ws, dv; self.stream = TS(pat)
    def next_batch(self, gt, sl, gas):
        prs = gt // (self.ws * gas) + 1
        chunk = self.stream.take(prs * self.ws)
        start = self.rank * prs; local = chunk[start:start+prs].to(dtype=I64)
        x, y = local[:-1].reshape(-1, sl), local[1:].reshape(-1, sl)
        return x.to(self.dv, non_blocking=True), y.to(self.dv, non_blocking=True)

class RN(M):
    def __init__(self, eps=None): super().__init__(); self.eps = eps
    def forward(self, x): return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class CL(nn.Linear):
    _qat_enabled: bool = False
    _qat_clip_pct: float = 0.9995
    def forward(self, x):
        w = self.weight.to(x.dtype)
        if CL._qat_enabled and self.training and w.ndim == 2:
            with NG():
                w32 = self.weight.float()
                row_clip = QT(w32.abs(), CL._qat_clip_pct, dim=1)
                scale = (row_clip / 31.0).clamp_min(1.0 / 31.0)
                w_q = (CLP(th.round(w32 / scale[:, None]), -31, 31) * scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()
        return F.linear(x, w, self.bias.to(x.dtype) if self.bias is not None else None)

def rf32(module):
    with NG():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CP)) and param.dtype != F32:
                param.data = param.data.float()

class RY(M):
    def __init__(self, dim, base=10000.0, tsl=1024, rd=0):
        super().__init__()
        self.dim = dim; self.base = base; self.tsl = tsl
        self.rope_dims = rd if rd > 0 else dim
        inv_freq = 1.0 / (base ** (AR(0, self.rope_dims, 2, dtype=F32) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0; self._cos_cached = self._sin_cached = None
    def forward(self, sl, dv, dtype):
        if self._cos_cached is None or self._seq_len_cached != sl or self._cos_cached.device != dv:
            rd = self.rope_dims
            if sl > self.tsl:
                scale = sl / self.tsl
                new_base = self.base * (scale ** (rd / (rd - 2)))
                inv_freq = 1.0 / (new_base ** (AR(0, rd, 2, dtype=F32, device=dv) / rd))
            else: inv_freq = self.inv_freq.to(dv)
            freqs = th.outer(AR(sl, device=dv, dtype=inv_freq.dtype), inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]; self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = sl
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)

def are(x, cos, sin, rd=0):
    if rd > 0 and rd < x.size(-1):
        x_rope, x_pass = x[..., :rd], x[..., rd:]
        half = rd // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = CAT((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return CAT((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return CAT((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

class CSA(M):
    def __init__(self, dim, nh, nkh, rb, qgi):
        super().__init__()
        self.num_heads, self.num_kv_heads = nh, nkh; self.head_dim = dim // nh
        kv_dim = nkh * self.head_dim
        self.c_q = CL(dim, dim, bias=False); self.c_k = CL(dim, kv_dim, bias=False)
        self.c_v = CL(dim, kv_dim, bias=False); self.proj = CL(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = P(FUL((nh,), qgi, dtype=F32))
        self.rope_dims = 0
        self.rotary = RY(self.head_dim, base=rb, tsl=1024)
        self.use_xsa = False
    def _xsa_efficient(self, y, v):
        B, T, H, D = y.shape; Hkv = v.size(-2); group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)
    def forward(self, x, v_embed=None, v_residual=None):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x)
        if v_embed is not None: v = v + v_embed
        if v_residual is not None: v = v + v_residual
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = are(q, cos, sin, self.rope_dims)
        k = are(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        if HAS_FA3:
            y = _fa3_func(q, k, v, causal=True)
            if isinstance(y, tuple): y = y[0]
        else:
            qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            y = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=None, is_causal=True,
                                               enable_gqa=(self.num_kv_heads != self.num_heads)).transpose(1, 2)
        if self.use_xsa: y = self._xsa_efficient(y, v)
        return self.proj(y.reshape(bsz, seqlen, dim))

class MLP(M):
    def __init__(self, dim, mm):
        super().__init__()
        self.fc = CL(dim, int(mm * dim), bias=False)
        self.proj = CL(int(mm * dim), dim, bias=False)
        self.proj._zero_init = True
    def forward(self, x):
        return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.5).square())

class SGT(M):
    def __init__(self, dim):
        super().__init__(); self.gate = P(Z(dim, dtype=F32))
    def forward(self, x):
        g = SG(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = CAT([ZL(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev

class BHE(M):
    def __init__(self, bgvs, bgd, dm):
        super().__init__()
        self.bgvs = bgvs
        self.embed = nn.Embedding(bgvs, bgd)
        NI.zeros_(self.embed.weight)
        self.proj = CL(bgd, dm, bias=False) if bgd != dm else None
        if self.proj is not None: NI.zeros_(self.proj.weight)
        self.scale = P(TT(0.05, dtype=F32))
    def bigram_hash(self, tokens):
        t = tokens.to(th.int32); mod = self.bgvs - 1
        out = EL(t); out[..., 0] = mod
        out[..., 1:] = th.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()
    def forward(self, token_ids):
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None: h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

class VE(M):
    def __init__(self, vs, ve_dim, kv_dim):
        super().__init__()
        self.embed = nn.Embedding(vs, ve_dim)
        NI.normal_(self.embed.weight, std=0.01)
        self.proj = CL(ve_dim, kv_dim, bias=False) if ve_dim != kv_dim else None
        if self.proj is not None: NI.zeros_(self.proj.weight)
        self.scale = P(TT(0.1, dtype=F32))
    def forward(self, token_ids):
        h = self.embed(token_ids)
        if self.proj is not None: h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

class BL(M):
    def __init__(self, dim, nh, nkh, mm, rb, qgi, layer_idx=0, ln_scale=False):
        super().__init__()
        self.attn_norm, self.mlp_norm = RN(), RN()
        self.attn = CSA(dim, nh, nkh, rb, qgi)
        self.mlp = MLP(dim, mm)
        self.attn_scale = P(ON(dim, dtype=F32))
        self.mlp_scale = P(ON(dim, dtype=F32))
        self.resid_mix = P(SK((ON(dim), Z(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
    def forward(self, x, x0, v_embed=None, v_residual=None):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor, v_embed=v_embed, v_residual=v_residual)
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor)
        return x_out

class GPT(M):
    def __init__(self, vs, nl, dm, nh, nkh,
                 mm, te, teis, lsc,
                 rb, qgi, se=True, be=True, backout_init=0.2,
                 bgvs=0, bgd=128, xsn=0,
                 rd=0, ln_scale=False,
                 vee=False, ved=128, vel="9,10"):
        super().__init__()
        self.te, self.teis = te, teis
        self.lsc = lsc
        self.se, self.be, self.num_layers = se, be, nl
        self.tok_emb = nn.Embedding(vs, dm)
        self.bigram = BHE(bgvs, bgd, dm) if bgvs > 0 else None
        self.smear = SGT(dm) if se else None
        self.backout_lambda = P(backout_init * ON(1)) if be else None
        self.num_encoder_layers = nl // 2
        self.num_decoder_layers = nl - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = P(ON(self.num_skip_weights, dm, dtype=F32))
        self.blocks = ML([
            BL(dm, nh, nkh, mm, rb, qgi,
                  layer_idx=i, ln_scale=ln_scale)
            for i in range(nl)
        ])
        if rd > 0:
            head_dim = dm // nh
            for block in self.blocks:
                block.attn.rope_dims = rd
                block.attn.rotary = RY(head_dim, base=rb, tsl=1024, rd=rd)
        if xsn > 0:
            for i in range(max(0, nl - xsn), nl):
                self.blocks[i].attn.use_xsa = True
        kv_dim = nkh * (dm // nh)
        self.ve_layer_indices = [int(x) for x in vel.split(",") if x.strip()] if vee else []
        if self.ve_layer_indices:
            self.ve_shared = VE(vs, ved, kv_dim)
            self.ve_layer_scales = PL([P(ON(1, dtype=F32)) for _ in self.ve_layer_indices])
        else:
            self.ve_shared = None; self.ve_layer_scales = PL()
        self.final_norm = RN()
        self.vrl_enabled = nl > 1
        if self.vrl_enabled:
            self.vrl_alphas = PL([
                P(TT(0.0, dtype=F32)) for _ in range(nl - 1)
            ])
        else:
            self.vrl_alphas = PL()
        self.lm_head = None if te else CL(dm, vs, bias=False)
        if self.lm_head is not None: self.lm_head._zero_init = True
        self._init_weights()
    def _init_weights(self):
        if self.te:
            NI.normal_(self.tok_emb.weight, mean=0.0, std=self.teis)
        nl = len(self.blocks)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False): NI.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    NI.orthogonal_(module.weight, gain=1.0)
                    if ".proj." in name or name.endswith(".proj"):
                        with NG(): module.weight.mul_(1.0 / math.sqrt(2 * nl))
        for i, block in enumerate(self.blocks):
            with NG():
                phase = SG(TT(3.0 * (i / max(nl-1, 1) - 0.5)))
                block.resid_mix.data[0] = phase * ON(block.resid_mix.shape[1])
                block.resid_mix.data[1] = (1-phase) * ON(block.resid_mix.shape[1])
    def _get_ve(self, layer_idx, ids, ve_cache):
        if self.ve_shared is None or layer_idx not in self.ve_layer_indices: return None
        if 've' not in ve_cache: ve_cache['ve'] = self.ve_shared(ids)
        ve_idx = self.ve_layer_indices.index(layer_idx)
        return ve_cache['ve'] * self.ve_layer_scales[ve_idx].to(dtype=ve_cache['ve'].dtype)
    def _run_layers(self, x, x0, ids):
        skips, backout_layer, x_backout = [], self.num_layers // 2, None
        ve_cache = {}
        v0_raw = None
        if self.vrl_enabled:
            blk0 = self.blocks[0]
            mix0 = blk0.resid_mix.to(dtype=x0.dtype)
            x_in0 = mix0[0][None, None, :] * x0 + mix0[1][None, None, :] * x0
            v0_raw = blk0.attn.c_v(blk0.attn_norm(x_in0) * blk0.ln_scale_factor)
        vrl_idx = 0
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, ids, ve_cache)
            v_res = None
            if i > 0 and v0_raw is not None:
                alpha = SG(self.vrl_alphas[vrl_idx].to(dtype=x.dtype))
                v_res = alpha * v0_raw
                vrl_idx += 1
            x = self.blocks[i](x, x0, v_embed=ve, v_residual=v_res); skips.append(x)
            if i == backout_layer: x_backout = x
        for i in range(self.num_decoder_layers):
            li = self.num_encoder_layers + i
            if skips: x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(li, ids, ve_cache)
            v_res = None
            if v0_raw is not None:
                alpha = SG(self.vrl_alphas[vrl_idx].to(dtype=x.dtype))
                v_res = alpha * v0_raw
                vrl_idx += 1
            x = self.blocks[li](x, x0, v_embed=ve, v_residual=v_res)
            if li == backout_layer and x_backout is None: x_backout = x
        if self.backout_lambda is not None and x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        return x
    def _embed(self, ids):
        x = self.tok_emb(ids)
        if self.bigram is not None: x = x + self.bigram(ids)
        x = F.rms_norm(x, (self.tok_emb.weight.shape[1],))
        if self.smear is not None: x = self.smear(x)
        return x
    def forward(self, ids, tgt):
        x0 = self._embed(ids); x = self._run_layers(x0, x0, ids)
        x_flat = self.final_norm(x).reshape(-1, x.size(-1)); targets = tgt.reshape(-1)
        logits_proj = F.linear(x_flat, self.tok_emb.weight) if self.te else self.lm_head(x_flat)
        logits = self.lsc * th.tanh(logits_proj / self.lsc)
        return F.cross_entropy(logits.float(), targets, reduction="mean")
    def forward_logits(self, ids):
        x0 = self._embed(ids); x = self.final_norm(self._run_layers(x0, x0, ids))
        logits = F.linear(x, self.tok_emb.weight.to(x.dtype)) if self.te else self.lm_head(x)
        return self.lsc * th.tanh(logits / self.lsc)

def ch(bm, tl, args, dv, gas, num_batches=256):
    hh = {}
    hooks = []
    param_to_name = {}
    for name, module in bm.named_modules():
        if isinstance(module, CL):
            pn = name + ".weight"
            param_to_name[id(module)] = pn
            cols = module.weight.shape[1]
            hh[pn] = Z(cols, cols, dtype=F32, device='cpu')
            def make_hook(mod_id, pname, ncols):
                count = [0]
                def hook_fn(module, input, output):
                    x = input[0].detach().float()
                    if x.ndim == 3:
                        x = x.reshape(-1, x.shape[-1])
                    xtx = (x.T @ x).cpu()
                    hh[pname] += xtx
                    count[0] += x.shape[0]
                return hook_fn
            h = module.register_forward_hook(make_hook(id(module), pn, cols))
            hooks.append(h)
    bm.eval()
    with IM(), AU():
        for _ in range(num_batches):
            x, y = tl.next_batch(args.tbt, args.tsl, gas)
            _ = bm(x, y)
    for h in hooks: h.remove()
    for name in hh:
        H = hh[name]
        H /= num_batches
        damp = 0.01 * DG(H).mean().clamp_min(1e-6)
        H += damp * EYE(H.shape[0])
        hh[name] = H
    bm.train()
    return hh

def evl(logits_fn, rank, ws, dv, vt,
                     bb, hs, ib,
                     ql, stride, ebs=256):
    total = vt.numel() - 1; windows, p = [], 0
    while p + ql <= total:
        s = 0 if p == 0 else (ql - stride); windows.append((p, s)); p += stride
    n = len(windows); per_rank = (n + ws - 1) // ws
    my_windows = windows[rank*per_rank:min((rank+1)*per_rank, n)]
    loss_sum = Z((), device=dv, dtype=F64)
    tok_count = Z((), device=dv, dtype=F64)
    byte_count = Z((), device=dv, dtype=F64)
    with IM():
        for i in range(0, len(my_windows), ebs):
            batch = my_windows[i:i+ebs]; bs = len(batch)
            x_list = [vt[w:w+ql] for w, _ in batch]
            y_list = [vt[w+1:w+ql+1] for w, _ in batch]
            pad = ebs - bs
            if pad > 0: x_list.extend([x_list[-1]]*pad); y_list.extend([y_list[-1]]*pad)
            x = SK(x_list).to(device=dv, dtype=I64)
            y = SK(y_list).to(device=dv, dtype=I64)
            with AU(): logits = logits_fn(x)
            for b in range(bs):
                s = batch[b][1]; sl, st = logits[b, s:], y[b, s:]
                loss_sum += F.cross_entropy(sl.float(), st, reduction="sum").to(F64)
                ns = st.numel(); tok_count += ns
                prev, tgt = x[b, s:s+ns], st
                tb = bb[tgt].to(I16)
                tb += (hs[tgt] & ~ib[prev]).to(I16)
                byte_count += tb.to(F64).sum()
    if IA() and II():
        for t in [loss_sum, tok_count, byte_count]: ARD(t, op=ROP.SUM)
    vl = (loss_sum / tok_count).item()
    return vl, vl / math.log(2.0) * (tok_count.item() / byte_count.item())

def rcp(spec: str) -> str:
    if not spec:
        return ""
    if spec == "1":
        return str(Path(EG("OUT_DIR", ".")) / "pre_export_model.pt")
    return spec

def mspc(path_spec: str, sd: dict[str, th.Tensor], log0):
    ckpt_path = rcp(path_spec)
    if not ckpt_path:
        return ""
    ckpt = {"model_state": {k: DC(v) for k, v in sd.items()}}
    path = Path(ckpt_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    th.save(ckpt, tmp_path)
    os.replace(tmp_path, path)
    return str(path)

def ree(a, bm, rank, ws, dv, dd, m0,
                    code, vt, bb, hs, ib, log0):
    cl = DTL(a.tf, rank, ws, dv)
    hh = ch(bm, cl, a, dv, 8 // ws,
                                num_batches=a.gcb)
    hm = {}
    for name, module in bm.named_modules():
        if isinstance(module, CL):
            sd_name = name + ".weight"
            h_name = name + ".weight"
            if h_name in hh:
                hm[sd_name] = hh[h_name]
    sd_cpu = {k: DC(v) for k, v in bm.state_dict().items()}
    cb = len(code.encode("utf-8")); sl = 16_000_000
    qr, qm = qsd(sd_cpu, hh=hm)
    if a.pp > 0:
        ai6 = []
        for name, info in qm.items():
            if mk(info) == "int6":
                qname = name + ".q"
                if qname in qr:
                    ai6.append(qr[qname].flatten().abs().float())
        if ai6:
            all_vals = CAT(ai6)
            k = max(1, int(a.pp * all_vals.numel()))
            threshold = all_vals.kthvalue(k).values.item()
            prc = 0
            for name, info in qm.items():
                if mk(info) == "int6":
                    qname = name + ".q"
                    if qname in qr:
                        mask = qr[qname].abs() <= int(threshold)
                        prc += mask.sum().item()
                        qr[qname][mask] = 0
            ti6 = sum(qr[n + ".q"].numel() for n, i in qm.items() if mk(i) == "int6" and n + ".q" in qr)
            log0(f"prune:{prc}/{ti6} ({100*prc/max(ti6,1):.1f}%) thr={threshold:.0f}")
    meta_blob, meta_names = eqm(qm)
    nti = {name: idx for idx, name in enumerate(meta_names)}
    parts = [PK("<I", len(meta_blob)), meta_blob]
    meta_bytes = 4 + len(meta_blob)
    thb = 0
    p6b = 0
    opb = 0
    p6t = 0
    tensor_order = sorted(qr.keys())
    for tname in tensor_order:
        t = qr[tname]
        bn = tname[:-2] if tname.endswith(".q") else ""
        pi = (
            tname.endswith(".q")
            and mk(qm.get(bn)) == "int6"
        )
        dtype_map = {I8: 0, F16: 1, F32: 2, BF: 3}
        dt = 5 if pi else dtype_map.get(t.dtype, 2)
        if pi:
            raw = pi6(t)
        else:
            t_np = t.contiguous().numpy() if t.dtype != BF else t.contiguous().view(U16).numpy()
            raw = t_np.tobytes()
        name_idx, suffix = etr(tname, nti)
        parts.append(PK("<HBBB", name_idx, suffix, dt, t.ndim))
        thb += 5 + 4 * t.ndim
        for d in t.shape: parts.append(PK("<I", d))
        parts.append(raw)
        if pi:
            p6b += len(raw)
            p6t += 1
        else:
            opb += len(raw)
    quant_raw = b"".join(parts)
    model_blob, model_codec_id = cmb(quant_raw)
    mb = len(model_blob); ts = cb + mb
    log0(
        "ab:"
        f" m={meta_bytes}"
        f" th={thb}"
        f" p6={p6b}"
        f" op={opb}"
        f" rt={len(quant_raw)}"
        f" cm={mb}"
        f" cd={mcn(model_codec_id)}"
        f" t6={p6t}"
    )
    log0(f"sz:m={mb} c={cb} t={ts}({ts/1e6:.2f}M)")
    if ts > sl: log0(f"warn:size {ts}+{ts - sl}")
    else: log0(f"size_ok:{ts/1e6:.2f} MB")
    if m0:
        with open("final_model.int6.ptz", "wb") as f: f.write(model_blob)
    if dd: BR()
    with open("final_model.int6.ptz", "rb") as f: model_blob_loaded = f.read()
    rd, _ = dmb(model_blob_loaded)
    o = 0
    meta_len = UF("<I", rd, o)[0]; o += 4
    lm, meta_names, ctr = dqm(rd[o:o+meta_len]); o += meta_len
    drm = {0: (I8, np.int8), 1: (F16, np.float16), 2: (F32, np.float32), 3: (BF, np.uint16)}
    lr = {}
    while o < len(rd):
        if ctr:
            name_idx, suffix, dt, ndim = UF("<HBBB", rd, o); o += 5
            tname = dtr(name_idx, suffix, meta_names)
        else:
            name_len = UF("<H", rd, o)[0]; o += 2
            tname = rd[o:o+name_len].decode("utf-8"); o += name_len
            dt, ndim = UF("<BB", rd, o); o += 2
        shape = []
        for _ in range(ndim):
            shape.append(UF("<I", rd, o)[0]); o += 4
        if dt == 4:
            numel = int(np.prod(shape, dtype=np.int64))
            nbytes = ((numel + 3) // 4) * 3
            raw = memoryview(rd)[o:o+nbytes]
            o += nbytes
            t = ui6l(raw, shape)
        elif dt == 5:
            numel = int(np.prod(shape, dtype=np.int64))
            nbytes = ((numel + 7) // 8) * 6
            raw = memoryview(rd)[o:o+nbytes]
            o += nbytes
            t = ui6(raw, shape)
        else:
            torch_dt, np_dt = drm[dt]
            numel = 1
            for s in shape: numel *= s
            nbytes = numel * np.dtype(np_dt).itemsize
            arr = np.frombuffer(rd, dtype=np_dt, count=numel, offset=o).copy()
            o += nbytes
            t = FN(arr).reshape(shape)
            if torch_dt == BF: t = t.view(BF)
        lr[tname] = t
    deq_state = dsd(lr, lm, sd_cpu)
    bm.load_state_dict(deq_state, strict=True)
    eval_sl = a.esl if a.esl > 0 else a.tsl
    vte = lvt(a.val_files, eval_sl) if eval_sl != a.tsl else vt
    rlf = CMP(bm.forward_logits, dynamic=False) if not bool(int(EG("TORCH_COMPILE_DISABLE", "0"))) else bm.forward_logits
    warmup_x = Z(a.ebs, eval_sl, dtype=I64, device=dv)
    bm.eval()
    with IM(), AU(): _ = rlf(warmup_x)
    SY(); t_eval = PC()
    q_vl, q_vb = evl(rlf, rank, ws, dv,
        vte, bb, hs, ib,
        eval_sl, a.evs, ebs=a.ebs)
    SY(); eval_time = PC() - t_eval
    log0(f"final_int6 val_loss:{q_vl:.4f} val_bpb:{q_vb:.4f} eval_time:{eval_time*1000:.0f}ms")
    log0(f"final_int6_exact val_loss:{q_vl:.8f} val_bpb:{q_vb:.8f}")
    if dd: DGP()

def main():
    global z5
    code = Path(__file__).read_text(encoding="utf-8"); a = H()
    z5 = CMP(z5)
    dd = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(EG("RANK", "0")); ws = int(EG("WORLD_SIZE", "1"))
    lrk = int(EG("LOCAL_RANK", "0"))
    if ws <= 0 or 8 % ws != 0: raise ER(f"bad WS={ws}")
    gas = 8 // ws; grad_scale = 1.0 / gas
    if not th.cuda.is_available(): raise RuntimeError("cuda req")
    dv = th.device(CU, lrk); th.cuda.set_device(dv)
    if dd: IGP(backend="nccl", device_id=dv); BR()
    m0 = rank == 0
    th.backends.cuda.matmul.allow_tf32 = True; th.backends.cudnn.allow_tf32 = True
    if not HAS_FA3:
        from th.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
        enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)
    lf = None
    if m0: os.makedirs("logs", exist_ok=True); lf = f"logs/{a.rid}.txt"; print(lf)
    def log0(msg, console=True):
        if not m0: return
        if console: print(msg)
        if lf:
            with open(lf, "a", encoding="utf-8") as f: print(msg, file=f)
    random.seed(a.sd); np.random.seed(a.sd); th.manual_seed(a.sd); th.cuda.manual_seed_all(a.sd)
    sp = spm.SentencePieceProcessor(model_file=a.tp)
    vt = lvt(a.vf, a.tsl)
    bb, hs, ib = bsl(sp, a.vs, dv)
    CL._qat_enabled = False
    CL._qat_clip_pct = a.qcp
    bm = GPT(
        vs=a.vs, nl=a.nl, dm=a.dm,
        nh=a.nh, nkh=a.nkh, mm=a.mm,
        te=a.te, teis=a.teis,
        lsc=a.lsc, rb=a.rb, qgi=a.qgi,
        se=a.se, be=a.be, backout_init=a.boi,
        bgvs=a.bgvs, bgd=a.bgd,
        xsn=a.xsn, rd=a.rd, ln_scale=a.lns,
        vee=a.vee, ved=a.ved, vel=a.vel,
    ).to(dv).bfloat16()
    for m in bm.modules():
        if isinstance(m, CL): m.float()
    rf32(bm)
    gs = bm.state_dict; ls = bm.load_state_dict
    cm = CMP(bm, dynamic=False, fullgraph=True) if not bool(int(EG("TORCH_COMPILE_DISABLE", "0"))) else bm
    model = DDP(cm, device_ids=[lrk], broadcast_buffers=False) if dd else cm
    eoc = rcp(a.eoc)
    if eoc:
        ckpt = th.load(eoc, map_location="cpu")
        ms = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        ls(ms, strict=True)
        ree(a, bm, rank, ws, dv, dd, m0,
                        code, vt, bb, hs, ib, log0)
        return
    bnp = list(bm.blocks.named_parameters())
    mpa = [p for n, p in bnp if p.ndim == 2 and not any(pat in n for pat in CP)]
    spa = [p for n, p in bnp if p.ndim < 2 or any(pat in n for pat in CP)]
    if bm.skip_weights.numel() > 0: spa.append(bm.skip_weights)
    if bm.smear is not None: spa.append(bm.smear.gate)
    if bm.backout_lambda is not None: spa.append(bm.backout_lambda)
    if bm.bigram is not None: spa.append(bm.bigram.scale)
    if bm.ve_shared is not None:
        spa.append(bm.ve_shared.scale)
        for s in bm.ve_layer_scales: spa.append(s)
    if bm.vrl_enabled:
        for alpha in bm.vrl_alphas: spa.append(alpha)
    tlr = a.telr if a.te else a.elr
    tpg = [{"params": [bm.tok_emb.weight], "lr": tlr, "base_lr": tlr}]
    if bm.bigram is not None:
        tpg.append({"params": [bm.bigram.embed.weight], "lr": tlr, "base_lr": tlr})
        if bm.bigram.proj is not None: mpa.append(bm.bigram.proj.weight)
    if bm.ve_shared is not None:
        tpg.append({"params": [bm.ve_shared.embed.weight], "lr": tlr, "base_lr": tlr})
        if bm.ve_shared.proj is not None: mpa.append(bm.ve_shared.proj.weight)
    ot = th.optim.AdamW(tpg, betas=(a.beta1, a.beta2), eps=a.aep, weight_decay=a.awd, fused=True)
    om = MU(mpa, lr=a.mlr, momentum=a.mum, ns_steps=a.mns, wd=a.mwd)
    for group in om.param_groups: group["base_lr"] = a.mlr
    os = th.optim.AdamW([{"params": spa, "lr": a.slr, "base_lr": a.slr}],
                                          betas=(a.beta1, a.beta2), eps=a.aep, weight_decay=a.awd, fused=True)
    opts = [ot, om, os]
    if bm.lm_head is not None:
        oh = th.optim.Adam([{"params": [bm.lm_head.weight], "lr": a.hlr, "base_lr": a.hlr}],
                                           betas=(a.beta1, a.beta2), eps=a.aep, fused=True)
        opts.insert(1, oh)
    tl = DTL(a.tf, rank, ws, dv)
    def zga():
        for opt in opts: opt.zero_grad(set_to_none=True)
    mwm = 1000.0 * a.mws if a.mws > 0 else None
    def lr_mul(step, ems):
        if a.wdi <= 0: return 1.0
        if mwm is None:
            wd0 = max(a.it - a.wdi, 0)
            return max((a.it - step) / max(a.wdi, 1), 0.0) if step >= wd0 else 1.0
        step_ms = ems / max(step, 1); wd_ms = a.wdi * step_ms
        rem_ms = max(mwm - ems, 0.0)
        return rem_ms / max(wd_ms, 1e-9) if rem_ms <= wd_ms else 1.0

    if a.wus > 0:
        ims = {n: DL(t) for n, t in gs().items()}
        ios = [copy.deepcopy(opt.state_dict()) for opt in opts]
        model.train()
        for wi in range(a.wus):
            zga()
            for ms in range(gas):
                if dd: model.require_backward_grad_sync = ms == gas - 1
                x, y = tl.next_batch(a.tbt, a.tsl, gas)
                with AU(): wl = model(x, y)
                (wl * grad_scale).backward()
            for opt in opts: opt.step()
            zga()
        ls(ims, strict=True)
        for opt, state in zip(opts, ios, strict=True): opt.load_state_dict(state)
        zga()
        if dd: model.require_backward_grad_sync = True
        tl = DTL(a.tf, rank, ws, dv)
    ema_state = {name: t.detach().float().clone() for name, t in gs().items()}
    ttms, sas = 0.0, None
    swa_state, swa_count = None, 0
    SY(); t0 = PC(); step = 0
    while True:
        last_step = step == a.it or (sas is not None and step >= sas)
        sv = last_step or (a.vle > 0 and step % a.vle == 0)
        if sv:
            SY(); ttms += 1000.0 * (PC() - t0)
            vl, vb = eval_val(a, model, rank, ws, dv, gas,
                              vt, bb, hs, ib)
            log0(f"step:{step}/{a.it} val_loss:{vl:.4f} val_bpb:{vb:.4f} train_time:{ttms:.0f}ms step_avg:{ttms/max(step,1):.2f}ms")
            SY(); t0 = PC()
        if last_step:
            if sas is not None and step < a.it:
                log0(f"stop:wall train:{ttms:.0f}ms step:{step}/{a.it}")
            break
        ems = ttms + 1000.0 * (PC() - t0)
        scale = lr_mul(step, ems)
        if a.lqt > 0 and scale < a.lqt and not CL._qat_enabled:
            CL._qat_enabled = True
        zga(); trl = Z((), device=dv)
        for ms in range(gas):
            if dd: model.require_backward_grad_sync = ms == gas - 1
            x, y = tl.next_batch(a.tbt, a.tsl, gas)
            with AU(): loss = model(x, y)
            trl += loss.detach(); (loss * grad_scale).backward()
        trl /= gas
        frac = min(step / a.mmws, 1.0) if a.mmws > 0 else 1.0
        for group in om.param_groups:
            group["momentum"] = (1-frac)*a.mmst + frac*a.mum
        for opt in opts:
            for group in opt.param_groups: group["lr"] = group["base_lr"] * scale
        if a.gcn > 0: th.nn.utils.clip_grad_norm_(bm.parameters(), a.gcn)
        for opt in opts: opt.step()
        zga()
        with NG():
            for name, t in gs().items():
                ema_state[name].mul_(a.ed).add_(t.detach().float(), alpha=1.0 - a.ed)
        step += 1
        ams = ttms + 1000.0 * (PC() - t0)
        if a.swe and scale < 0.2 and step % a.swi == 0:
            if swa_state is None:
                swa_state = {n: DL(t) for n, t in gs().items()}
                swa_count = 1
            else:
                for n, t in gs().items(): swa_state[n] += DC(t)
                swa_count += 1
        if a.tle > 0 and (step <= 10 or step % a.tle == 0):
            log0(f"step:{step}/{a.it} train_loss:{trl.item():.4f} train_time:{ams:.0f}ms step_avg:{ams/step:.2f}ms")
        rc = mwm is not None and ams >= mwm
        if dd and mwm is not None:
            rct = TT(int(rc), device=dv); ARD(rct, op=ROP.MAX); rc = bool(rct.item())
        if sas is None and rc: sas = step
    cs = gs()
    avg_state = {name: t.to(dtype=cs[name].dtype) for name, t in ema_state.items()}
    ls(avg_state, strict=True)
    mspc(a.spc, gs(), log0)
    ree(a, bm, rank, ws, dv, dd, m0,
                    code, vt, bb, hs, ib, log0)

if __name__ == "__main__":
    main()
