from __future__ import annotations
import copy, glob, io, json, lzma, math, os, random, struct, subprocess, sys, time, uuid, zlib
try:
    import zstandard as zstd; HZ = True
except ImportError: HZ = False
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch as th, torch.distributed as dist, torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
EG=os.environ.get; GI=lambda k,d=0:int(EG(k,d)); GF=lambda k,d=0:float(EG(k,d)); GB=lambda k,d=0:bool(int(EG(k,d))); JP=os.path.join; PC=time.perf_counter
BF=th.bfloat16; F16=th.float16; F32=th.float32; F64=th.float64; I8=th.int8; I16=th.int16; I64=th.int64; BO=th.bool; U16=th.uint16
AC=th.autocast; IM=th.inference_mode; Z=th.zeros; TT=th.tensor; QT=th.quantile; EM=th.empty; FN=th.from_numpy; SK=th.stack; PK=struct.pack; UF=struct.unpack_from; ER=ValueError; FE=FileNotFoundError; RE=RuntimeError; CU="cuda"; AU=lambda e=1:AC(device_type=CU,dtype=BF,enabled=e); CG=lambda t:t.contiguous(); DC=lambda t:t.detach().cpu(); DL=lambda t:DC(t).clone(); DX=lambda t:CG(DC(t)); AS=lambda x,d:x.astype(d,copy=False); NM=lambda s:int(np.prod(s,dtype=np.int64)); FB=np.frombuffer; NE=np.empty; NZ=np.zeros; RM=F.rms_norm; LI=F.linear; PT=Path; GG=glob.glob
P=nn.Parameter; PL=nn.ParameterList; M=nn.Module; ML=nn.ModuleList; NI=nn.init; NG=th.no_grad; CLP=th.clamp; DG=th.diag; CMP=th.compile; SY=th.cuda.synchronize
IA=dist.is_available; II=dist.is_initialized; BR=dist.barrier; GWS=dist.get_world_size; GRK=dist.get_rank; IGP=dist.init_process_group; DGP=dist.destroy_process_group; ARD=dist.all_reduce; ROP=dist.ReduceOp
ZL=th.zeros_like; EL=th.empty_like; AR=th.arange; SG=th.sigmoid; CAT=th.cat; ON=th.ones; FUL=th.full; EYE=th.eye
TF=lambda t:t.float(); RS=lambda t,*s:t.reshape(*s); TY=lambda t,d:t.to(dtype=d); IT=lambda t:t.item()
N8=np.uint8; N6=np.int16; N1=np.int8; N4=np.uint32; NB=1
U="utf-8"; FM="final_model.int6.ptz"; MS="model_state"; MB="momentum_buffer"; TD="TORCH_COMPILE_DISABLE"; PA="params"; BL="base_lr"; QS=".q"; SS=".scale"; WT=".k"; F4="<I"; F7="<H"; F2="<BB"; F3="<HB"; F5="<HBBB"; F6="<BBBB"; LT="little"
try:
    from flash_attn_interface import flash_attn_func as _fa3_func
    HAS_FA3 = True
except ImportError:
    HAS_FA3 = False

class H:
    dp=EG("DP","./data/datasets/fineweb10B_sp1024"); tf=JP(dp,"fineweb_train_*.bin"); vf=JP(dp,"fineweb_val_*.bin")
    tp=EG("TP","./data/tokenizers/fineweb_1024_bpe.model"); rid=EG("RUN_ID",str(uuid.uuid4())); sd=GI("SEED",42)
    vbs=GI("VBS",524_288); vle=GI("VLE",500); tle=GI("TLE",200); it=GI("IT",20000)
    wus=GI("WUS",20); tbt=GI("TBT",786_432); tsl=GI("TSL",2048); esl=GI("ESL",2048)
    evs=GI("EVS",64); ebs=GI("EBS",256); mws=GF("MWS",600.0); qgi=GF("QGI",1.5)
    vs=GI("VS",1024); nl=GI("NL",11); nkh=GI("NKH",4); dm=GI("DM",512); nh=GI("NH",8); mm=GI("MM",3); te=GB("TE",1)
    rb=GF("RB",10000.0); lsc=GF("LSC",30.0); elr=GF("ELR",0.6); hlr=GF("HLR",0.008)
    telr=GF("TELR",0.035); teis=GF("TEIS",0.005); mlr=GF("MLR",0.025); slr=GF("SLR",0.025)
    mum=GF("MUM",0.99); mns=GI("MNS",5); mmst=GF("MMST",0.92); mmws=GI("MMWS",1500)
    mwd=GF("MWD",0.04); beta1=GF("B1",0.9); beta2=GF("B2",0.95); aep=GF("AEP",1e-8)
    awd=GF("AWD",0.04); gcn=GF("GCN",0.3); se=GB("SE",1); be=GB("BE",0); boi=GF("BOI",0.2); ed=GF("ED",0.997)
    swe=GB("SWE",1); swi=GI("SWI",50); wdi=GI("WDI",3500); lqt=GF("LQT",0.15)
    bgvs=GI("BGVS",2048); bgd=GI("BGD",128); xsn=GI("XSN",11); rd=GI("RD",16)
    lns=GB("LNS",1); vee=GB("VEE",1); ved=GI("VED",128); vel=EG("VEL","9,10")
    gcb=GI("GCB",256); gbs=GI("GBS",128); qcp=GF("QCP",0.9995); pp=GF("PP",0.02); spc=EG("SPC",""); eoc=EG("EOC","")

def z5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16(); X /= X.norm() + eps
    tr = G.size(0) > G.size(1)
    if tr: X = X.T
    for _ in range(steps):
        A = X @ X.T; B = b * A + c * A @ A; X = a * X + B @ X
    return X.T if tr else X

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
            params = group[PA]
            if not params: continue
            lr, momentum, ns_steps = group["lr"], group["momentum"], group["ns_steps"]
            nesterov = group["nesterov"]
            total_params = sum(int(p.numel()) for p in params)
            uf = Z(total_params, device=params[0].device, dtype=BF)
            curr = 0
            for i, p in enumerate(params):
                if i % ws == rk and p.grad is not None:
                    g = p.grad; state = self.state[p]
                    if MB not in state: state[MB] = ZL(g)
                    buf = state[MB]; buf.mul_(momentum).add_(g)
                    if nesterov: g = g.add(buf, alpha=momentum)
                    g = z5(g, steps=ns_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    uf[curr : curr + p.numel()] = RS(g, -1)
                curr += p.numel()
            if dd: ARD(uf, op=ROP.SUM)
            wd = group.get("wd", 0.0); curr = 0
            for p in params:
                g = TY(uf[curr : curr + p.numel()].view_as(p), p.dtype)
                if wd > 0: p.data.mul_(1.0 - lr * wd)
                p.add_(g, alpha=-lr); curr += p.numel()
        return loss

def bsl(sp, vs, dv):
    sv = int(sp.vocab_size()); ts = max(sv, vs)
    bbn = NZ((ts,), dtype=N6)
    hln = NZ((ts,), dtype=np.bool_)
    ibn = np.ones((ts,), dtype=np.bool_)
    for tid in range(sv):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid): continue
        ibn[tid] = False
        if sp.is_byte(tid): bbn[tid] = 1; continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"): hln[tid] = True; piece = piece[1:]
        bbn[tid] = len(piece.encode(U))
    return (TT(bbn, dtype=I16, device=dv),
            TT(hln, dtype=BO, device=dv),
            TT(ibn, dtype=BO, device=dv))

def lv(pat, sl):
    fs = [PT(p) for p in sorted(GG(pat))]
    if not fs: raise FE(f"no:{pat}")
    ts = CG(CAT([lds(f) for f in fs]))
    u = ((ts.numel() - 1) // sl) * sl
    if u <= 0: raise ER(f"val<{sl}")
    return ts[:u + 1]

def vv(a, model, rk, ws, dv, gas,
             vt, bb, hs, ib, esl=0):
    sl = esl if esl > 0 else a.tsl
    lbs = a.vbs // (ws * gas) // sl
    tsq = (vt.numel() - 1) // sl
    ss = (tsq * rk) // ws; se = (tsq * (rk + 1)) // ws
    vls = Z((), device=dv, dtype=F64)
    vtc = Z((), device=dv, dtype=F64)
    vbc = Z((), device=dv, dtype=F64)
    model.eval()
    with IM():
        for bss in range(ss, se, lbs):
            bse = min(bss + lbs, se)
            lc = vt[bss*sl:(bse*sl)+1].to(device=dv, dtype=I64, non_blocking=NB)
            x, y = RS(lc[:-1], -1, sl), RS(lc[1:], -1, sl)
            with AU():
                bl = model(x, y).detach()
            vls += bl.to(F64) * float(y.numel())
            vtc += float(y.numel())
            tb = TY(bb[RS(y, -1)], I16)
            tb += TY(hs[RS(y, -1)] & ~ib[RS(x, -1)], I16)
            vbc += tb.to(F64).sum()
    if IA() and II():
        for t in [vls, vtc, vbc]: ARD(t, op=ROP.SUM)
    vl = vls / vtc
    bpt = IT(vl) / math.log(2.0); tpb = IT(vtc) / IT(vbc)
    model.train(); return float(IT(vl)), float(bpt * tpb)

CP = tuple(
    p for p in "u,t,r,g,w,s.g,o,g.s,y,v.s".split(",") if p)
IC=lambda n:any(n==p or n.startswith(p+".") or n.endswith("."+p) or f".{p}." in n for p in CP)
I8K = 65_536
I8D = F16
MP = "p"
MC = "c"
M6 = 6
M8 = 8
Q0 = b"QMB1"
Q1 = b"QMB2"
QMB = b"QMB3"
QCB = b"QCB1"
KZ = 1
KG = 2
KL = 3
LF = [{
    "id": lzma.FILTER_LZMA2,
    "dict_size": 17<<20,
    "lc": 0,
    "lp": 1,
    "pb": 0,
    "mode": lzma.MODE_NORMAL,
    "nice_len": 64,
    "mf": lzma.MF_HC3,
}]

def cpm(name):
    if name.startswith("x.") or name.startswith("h."): return "e"
    if ".m." in name: return "m"
    if name.startswith("g."): return "b"
    if ".a." in name or (".p." in name and ".m." not in name): return "a"
    if name.startswith("v."): return "v"
    return "o"

def mk(info):
    if info == MP or info == "passthrough":
        return "p"
    if info == MC or info == "passthrough_ctrl":
        return "c"
    if info == M6 or (isinstance(info, dict) and info.get("type") == "int6"):
        return "6"
    if info == M8 or (isinstance(info, dict) and info.get("type") == "int8"):
        return "8"
    return None

def eq(meta: dict[str, object]) -> tuple[bytes, list[str]]:
    names = sorted(meta)
    km = {
        MP: 0,
        MC: 1,
        M6: 2,
        M8: 3,
        "passthrough": 0,
        "passthrough_ctrl": 1,
    }
    m2 = len(names) < 256 and max((len(name) for name in names), default=0) < 256
    parts = [Q1 if m2 else Q0, PK("<B" if m2 else "<H", len(names))]
    for name in names:
        kind = meta[name]
        kc = km.get(kind)
        if kc is None:
            kn = mk(kind)
            if kn == "6": kc = 2
            elif kn == "8": kc = 3
            else: raise ER(f"bad meta {name}:{kind!r}")
        xb = name.encode(U)
        parts.append(PK(F2 if m2 else F3, len(xb), kc))
        parts.append(xb)
    return b"".join(parts), names

def dq(blob: bytes) -> tuple[dict[str, object], list[str], int]:
    q = blob[:4]
    if q in (Q0, Q1, QMB):
        c = q != Q0; o = 4
        ec = UF("<B" if c else "<H", blob, o)[0]; o += 2 - c
        km = {0: MP, 1: MC, 2: M6, 3: M8}
        f, s = (F2, 2) if c else (F3, 3)
        meta, names = {}, []
        for _ in range(ec):
            nl, kc = UF(f, blob, o); o += s
            name = blob[o:o+nl].decode(U); o += nl
            meta[name] = km[kc]; names.append(name)
        return meta, names, 1 + c + (q == QMB)
    return json.loads(blob.decode(U)), [], 0

def tr(tname: str, nti: dict[str, int]) -> tuple[int, int]:
    if tname in nti:
        return nti[tname], 0
    if tname.endswith(QS) and tname[:-2] in nti:
        bn, suffix = tname[:-2], 1
    elif tname.endswith(SS) and tname[:-6] in nti:
        bn, suffix = tname[:-6], 2
    else:
        bn, suffix = tname, 0
    return nti[bn], suffix

def rt(ni: int, suffix: int, names: list[str]) -> str:
    bn = names[ni]
    if suffix == 1: return bn + QS
    if suffix == 2: return bn + SS
    return bn

def qg(weight, hessian=None, cr=31, block_size=128):
    t32 = TF(weight)
    if t32.ndim != 2 or hessian is None:
        return qp(t32, cr)
    rows, cols = t32.shape
    H = TF(hessian).clone()
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
        return qp(t32, cr)
    best_q = None; best_s = None; best_err = float('inf')
    for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
        if pct < 1.0:
            rc = QT(t32.abs(), pct, dim=1)
        else:
            rc = t32.abs().amax(dim=1)
        s = (rc / cr).clamp_min(1.0 / cr).to(F16)
        sf = TF(s)
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
                err = (w - TF(q) * sf) / d
                W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
                Err1[:, i] = err
            Q[:, i1:i2] = Q1
            if i2 < cols:
                W_work[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
        recon = TF(Q) * sf[:, None]
        mse = IT((W - recon).pow(2).mean())
        if mse < best_err:
            best_q, best_s, best_err = Q, s, mse
    best_q = best_q[:, inv_perm]
    return best_q, best_s

def qp(t32, cr=31):
    if t32.ndim == 2:
        best_q, best_s, best_err = None, None, float('inf')
        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
            if pct < 1.0:
                rc = QT(t32.abs(), pct, dim=1)
            else:
                rc = t32.abs().amax(dim=1)
            s = (rc / cr).clamp_min(1.0 / cr).to(F16)
            q = CLP(th.round(t32 / TF(s)[:, None]), -cr, cr).to(I8)
            recon = TF(q) * TF(s)[:, None]
            err = IT((t32 - recon).pow(2).mean())
            if err < best_err:
                best_q, best_s, best_err = q, s, err
        return best_q, best_s
    amax = IT(t32.abs().max())
    scale = TT(amax / cr if amax > 0 else 1.0, dtype=F16)
    q = CLP(th.round(t32 / TF(scale)), -cr, cr).to(I8)
    return q, scale

def q8(t):
    t32 = TF(t)
    if t32.ndim == 2:
        clip_q = 99.99984 / 100.0
        ca = QT(t32.abs(), clip_q, dim=1) if t32.numel() else EM((t32.shape[0],), dtype=F32)
        ct = th.maximum(th.minimum(t32, ca[:, None]), -ca[:, None])
        scale = (ca / 127.0).clamp_min(1.0 / 127.0)
        q = CG(CLP(th.round(ct / scale[:, None]), -127, 127).to(I8))
        return q, CG(TY(scale, I8D))
    clip_q = 99.99984 / 100.0
    ca = float(IT(QT(t32.abs().flatten(), clip_q))) if t32.numel() else 0.0
    scale = TT(ca / 127.0 if ca > 0 else 1.0, dtype=F32)
    q = CG(CLP(th.round(CLP(t32, -ca, ca) / scale), -127, 127).to(I8))
    return q, scale

def qs(sd, hh=None):
    r, m = {}, {}
    i6c = "mabv"
    for name, tn in sd.items():
        t = DX(tn)
        cat = cpm(name)
        if not t.is_floating_point() or t.numel() <= I8K:
            r[name] = t.to(F16) if t.is_floating_point() else t
            m[name] = MP; continue
        if IC(name):
            r[name] = TF(t); m[name] = MC; continue
        if cat in i6c and t.ndim >= 1:
            H = hh.get(name) if hh else None
            q, s = qg(t, hessian=H)
            r[name + QS] = q; r[name + SS] = s
            m[name] = M6; continue
        q, s = q8(t)
        r[name + QS] = q; r[name + SS] = s
        m[name] = M8
    return r, m

def ds(result, meta, tsd):
    out = {}
    for name, orig in tsd.items():
        info = meta.get(name)
        if info is None: continue
        od = orig.dtype
        if info in (MP, MC) or mk(info) in "pc":
            t = result[name]
            if t.dtype == F16 and od in (F32, BF): t = t.to(od)
            out[name] = t; continue
        q, s = result[name + QS], result[name + SS]
        if s.ndim > 0:
            out[name] = (TF(q) * RS(TF(s), q.shape[0], *([1]*(q.ndim-1)))).to(od)
        else:
            out[name] = (TF(q) * float(IT(s))).to(od)
    return out

def ze6(arr_i16: np.ndarray) -> np.ndarray:
    arr = AS(arr_i16, N6)
    if arr.size == 0:
        return NE((0,), dtype=N8)
    if arr.min() < -31 or arr.max() > 31:
        raise ER("int6 range")
    out = np.where(arr >= 0, arr * 2, (-arr) * 2 - 1)
    return AS(out, N8)

def zd6(u8: np.ndarray) -> np.ndarray:
    u = AS(u8, N6)
    return AS(np.where((u & 1) == 0, u // 2, -((u + 1) // 2)), N6)

def p6l(t: Tensor) -> bytes:
    q = DX(t)
    if q.dtype != I8:
        raise TypeError(f"need int8, got {q.dtype}")
    arr = AS(RS(q.numpy(), -1), N6)
    if arr.size == 0:
        return b""
    if arr.min() < -31 or arr.max() > 31:
        raise ER("int6 range")
    u = AS(arr + 31, N8)
    pad = (-u.size) % 4
    if pad:
        u = np.pad(u, (0, pad), constant_values=0)
    groups = AS(RS(u, -1, 4), N4)
    packed = (groups[:, 0]
              | (groups[:, 1] << 6)
              | (groups[:, 2] << 12)
              | (groups[:, 3] << 18))
    out = NE(packed.size * 3, dtype=N8)
    out[0::3] = (packed & 0xFF).astype(N8)
    out[1::3] = ((packed >> 8) & 0xFF).astype(N8)
    out[2::3] = ((packed >> 16) & 0xFF).astype(N8)
    return out.tobytes()

def u6l(raw: bytes | memoryview, shape: list[int]) -> Tensor:
    numel = NM(shape)
    if numel == 0:
        return EM(shape, dtype=I8)
    pu = FB(raw, dtype=N8)
    groups = (numel + 3) // 4
    if pu.size != groups * 3:
        raise ER(f"bad int6 sz {groups*3}!={pu.size}")
    tr = AS(RS(pu, -1, 3), N4)
    packed = tr[:, 0] | (tr[:, 1] << 8) | (tr[:, 2] << 16)
    u = NE(groups * 4, dtype=N8)
    u[0::4] = (packed & 0x3F).astype(N8)
    u[1::4] = ((packed >> 6) & 0x3F).astype(N8)
    u[2::4] = ((packed >> 12) & 0x3F).astype(N8)
    u[3::4] = ((packed >> 18) & 0x3F).astype(N8)
    arr = AS(u[:numel], N6) - 31
    return FN(RS(AS(arr, N1), shape))

def p6(t: Tensor) -> bytes:
    q = DX(t)
    if q.dtype != I8:
        raise TypeError(f"need int8, got {q.dtype}")
    arr = AS(RS(q.numpy(), -1), N6)
    if arr.size == 0:
        return b""
    u = ze6(AS(arr, N6))
    numel = u.size
    pad = (-numel) % 8
    if pad:
        u = np.pad(u, (0, pad), constant_values=0)
    out = bytearray()
    for bit in range(6):
        bits = RS(AS((u >> bit) & 1, N8), -1, 8)
        out.extend(RS(np.packbits(bits, axis=1, bitorder=LT), -1).tobytes())
    return bytes(out)

def u6(raw: bytes | memoryview, shape: list[int]) -> Tensor:
    numel = NM(shape)
    if numel == 0:
        return EM(shape, dtype=I8)
    padded = ((numel + 7) // 8) * 8
    pb = padded // 8
    pu = FB(raw, dtype=N8)
    expected = pb * 6
    if pu.size != expected:
        raise ER(f"bad int6 sz {expected}!={pu.size}")
    u = NZ(padded, dtype=N8)
    o = 0
    for bit in range(6):
        plane = pu[o:o + pb]
        o += pb
        bits = np.unpackbits(plane, bitorder=LT)
        u |= (AS(bits, N8) << bit)
    arr = zd6(u[:numel])
    return FN(RS(AS(arr, N1), shape))

def cn(cid: int) -> str:
    if cid == KZ:
        return "s"
    if cid == KG:
        return "z"
    if cid == KL:
        return "l"
    return f"u{cid}"

def cm(raw: bytes) -> tuple[bytes, int]:
    candidates = [
        (KL, lzma.compress(raw, format=lzma.FORMAT_RAW, filters=LF)),
        (KG, zlib.compress(raw, level=9)),
    ]
    if HZ:
        candidates.append((KZ, zstd.ZstdCompressor(level=19).compress(raw)))
    cid, payload = min(candidates, key=lambda item: len(item[1]))
    return QCB + bytes([cid]) + payload, cid

def db(blob: bytes) -> tuple[bytes, str]:
    if blob.startswith(QCB):
        cid = blob[len(QCB)]
        payload = blob[len(QCB) + 1:]
        if cid == KZ:
            if not HZ:
                raise RE("zstd unavailable")
            return zstd.ZstdDecompressor().decompress(payload), cn(cid)
        if cid == KG:
            return zlib.decompress(payload), cn(cid)
        if cid == KL:
            try:
                return lzma.decompress(payload, format=lzma.FORMAT_RAW, filters=LF), cn(cid)
            except lzma.LZMAError:
                return lzma.decompress(payload), "x"
        raise ER(f"bad codec {cid}")
    if HZ:
        try:
            return zstd.ZstdDecompressor().decompress(blob), "zs"
        except Exception:
            pass
    return zlib.decompress(blob), "z9"

def lds(file):
    hb = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1: raise ER(f"bad hdr {file}")
    nt = int(header[2])
    if file.stat().st_size != hb + nt * np.dtype("<u2").itemsize: raise ER(f"bad size {file}")
    return FN(AS(np.fromfile(file, dtype="<u2", count=nt, offset=hb), np.uint16))

class TS:
    def __init__(self, pat):
        self.fs = [PT(p) for p in sorted(GG(pat))]
        if not self.fs: raise FE(f"no:{pat}")
        self.fi = 0; self.ts = lds(self.fs[0]); self.p = 0
    def af(self):
        self.fi = (self.fi + 1) % len(self.fs); self.ts = lds(self.fs[self.fi]); self.p = 0
    def tk(self, n):
        chunks, r = [], n
        while r > 0:
            avail = self.ts.numel() - self.p
            if avail <= 0: self.af(); continue
            k = min(r, avail); chunks.append(self.ts[self.p:self.p+k]); self.p += k; r -= k
        return chunks[0] if len(chunks) == 1 else CAT(chunks)

class DTL:
    def __init__(self, pat, rk, ws, dv):
        self.rk, self.ws, self.dv = rk, ws, dv; self.s = TS(pat)
    def nb(self, gt, sl, gas):
        prs = gt // (self.ws * gas) + 1
        ck = self.s.tk(prs * self.ws)
        st = self.rk * prs; lc = TY(ck[st:st+prs], I64)
        x, y = RS(lc[:-1], -1, sl), RS(lc[1:], -1, sl)
        return x.to(self.dv, non_blocking=NB), y.to(self.dv, non_blocking=NB)

class RN(M):
    def __init__(self, eps=None): super().__init__(); self.e = eps
    def forward(self, x): return RM(x, (x.size(-1),), eps=self.e)

class CL(nn.Linear):
    _qat_enabled: bool = False
    _qat_clip_pct: float = 0.9995
    def forward(self, x):
        w = self.weight.to(x.dtype)
        if CL._qat_enabled and self.training and w.ndim == 2:
            with NG():
                w32 = TF(self.weight)
                row_clip = QT(w32.abs(), CL._qat_clip_pct, dim=1)
                scale = (row_clip / 31.0).clamp_min(1.0 / 31.0)
                w_q = (CLP(th.round(w32 / scale[:, None]), -31, 31) * scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()
        return LI(x, w, self.bias.to(x.dtype) if self.bias is not None else None)

def rf32(module):
    with NG():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or IC(name)) and param.dtype != F32:
                param.data = TF(param.data)

class RY(M):
    def __init__(self, dim, base=10000.0, tsl=1024, rd=0):
        super().__init__()
        self.b = base; self.t = tsl
        self.rd = rd if rd > 0 else dim
        inv_freq = 1.0 / (base ** (AR(0, self.rd, 2, dtype=F32) / self.rd))
        self.register_buffer("ifq", inv_freq, persistent=False)
        self.slc = 0; self.cc = self.sc = None
    def forward(self, sl, dv, dtype):
        if self.cc is None or self.slc != sl or self.cc.device != dv:
            rd = self.rd
            if sl > self.t:
                scale = sl / self.t
                new_base = self.b * (scale ** (rd / (rd - 2)))
                inv_freq = 1.0 / (new_base ** (AR(0, rd, 2, dtype=F32, device=dv) / rd))
            else: inv_freq = self.ifq.to(dv)
            freqs = th.outer(AR(sl, device=dv, dtype=inv_freq.dtype), inv_freq)
            self.cc = freqs.cos()[None, :, None, :]; self.sc = freqs.sin()[None, :, None, :]
            self.slc = sl
        return TY(self.cc, dtype), TY(self.sc, dtype)

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
        self.nh, self.nkh = nh, nkh; self.hd = dim // nh
        kv_dim = nkh * self.hd
        self.q = CL(dim, dim, bias=False); self.k = CL(dim, kv_dim, bias=False)
        self.v = CL(dim, kv_dim, bias=False); self.p = CL(dim, dim, bias=False)
        self.p._zero_init = True
        self.g = P(FUL((nh,), qgi, dtype=F32))
        self.rd = 0
        self.ro = RY(self.hd, base=rb, tsl=1024)
        self.ux = False
    def xe(self, y, v):
        B, T, H, D = y.shape; Hkv = v.size(-2); group = H // Hkv
        y_g = RS(y, B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return RS(y_g - proj, B, T, H, D)
    def forward(self, x, ve=None, vr=None):
        bsz, seqlen, dim = x.shape
        q = RS(self.q(x), bsz, seqlen, self.nh, self.hd)
        k = RS(self.k(x), bsz, seqlen, self.nkh, self.hd)
        v = self.v(x)
        if ve is not None: v = v + ve
        if vr is not None: v = v + vr
        v = RS(v, bsz, seqlen, self.nkh, self.hd)
        q, k = RM(q, (q.size(-1),)), RM(k, (k.size(-1),))
        cos, sin = self.ro(seqlen, x.device, q.dtype)
        q = are(q, cos, sin, self.rd)
        k = are(k, cos, sin, self.rd)
        q = q * TY(self.g, q.dtype)[None, None, :, None]
        if HAS_FA3:
            y = _fa3_func(q, k, v, causal=True)
            if isinstance(y, tuple): y = y[0]
        else:
            qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            y = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=None, is_causal=True,
                                               enable_gqa=(self.nkh != self.nh)).transpose(1, 2)
        if self.ux: y = self.xe(y, v)
        return self.p(RS(y, bsz, seqlen, dim))

class MLP(M):
    def __init__(self, dim, mm):
        super().__init__()
        self.f = CL(dim, int(mm * dim), bias=False)
        self.p = CL(int(mm * dim), dim, bias=False)
        self.p._zero_init = True
    def forward(self, x):
        return self.p(F.leaky_relu(self.f(x), negative_slope=0.5).square())

class SGT(M):
    def __init__(self, dim):
        super().__init__(); self.g = P(Z(dim, dtype=F32))
    def forward(self, x):
        g = SG(TY(self.g, x.dtype))[None, None, :]
        x_prev = CAT([ZL(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev

class BHE(M):
    def __init__(self, bgvs, bgd, dm):
        super().__init__()
        self.bv = bgvs
        self.e = nn.Embedding(bgvs, bgd)
        NI.zeros_(self.e.weight)
        self.p = CL(bgd, dm, bias=False) if bgd != dm else None
        if self.p is not None: NI.zeros_(self.p.weight)
        self.s = P(TT(0.05, dtype=F32))
    def bh(self, tokens):
        t = tokens.to(th.int32); mod = self.bv - 1
        out = EL(t); out[..., 0] = mod
        out[..., 1:] = th.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()
    def forward(self, ids):
        h = self.e(self.bh(ids))
        if self.p is not None: h = self.p(h)
        return h * TY(self.s, h.dtype)

class VE(M):
    def __init__(self, vs, ve_dim, kv_dim):
        super().__init__()
        self.e = nn.Embedding(vs, ve_dim)
        NI.normal_(self.e.weight, std=0.01)
        self.p = CL(ve_dim, kv_dim, bias=False) if ve_dim != kv_dim else None
        if self.p is not None: NI.zeros_(self.p.weight)
        self.s = P(TT(0.1, dtype=F32))
    def forward(self, ids):
        h = self.e(ids)
        if self.p is not None: h = self.p(h)
        return h * TY(self.s, h.dtype)

class BL(M):
    def __init__(self, dim, nh, nkh, mm, rb, qgi, li=0, ln_scale=False):
        super().__init__()
        self.an, self.mn = RN(), RN()
        self.a = CSA(dim, nh, nkh, rb, qgi)
        self.m = MLP(dim, mm)
        self.u = P(ON(dim, dtype=F32))
        self.t = P(ON(dim, dtype=F32))
        self.r = P(TF(SK((ON(dim), Z(dim)))))
        self.lsf = 1.0 / math.sqrt(li + 1) if ln_scale else 1.0
    def forward(self, x, x0, ve=None, vr=None):
        mix = TY(self.r, x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.a(self.an(x_in) * self.lsf, ve=ve, vr=vr)
        x_out = x_in + TY(self.u, x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + TY(self.t, x_out.dtype)[None, None, :] * self.m(self.mn(x_out) * self.lsf)
        return x_out

class GPT(M):
    def __init__(self, vs, nl, dm, nh, nkh,
                 mm, te, teis, lsc,
                 rb, qgi, se=True, be=True, backout_init=0.2,
                 bgvs=0, bgd=128, xsn=0,
                 rd=0, ln_scale=False,
                 vee=False, ved=128, vel="9,10"):
        super().__init__()
        self.t, self.ts = te, teis
        self.l = lsc
        self.nl = nl
        self.x = nn.Embedding(vs, dm)
        self.g = BHE(bgvs, bgd, dm) if bgvs > 0 else None
        self.s = SGT(dm) if se else None
        self.o = P(backout_init * ON(1)) if be else None
        self.nel = nl // 2
        self.ndl = nl - self.nel
        self.nsw = min(self.nel, self.ndl)
        self.w = P(ON(self.nsw, dm, dtype=F32))
        self.b = ML([
            BL(dm, nh, nkh, mm, rb, qgi,
                  li=i, ln_scale=ln_scale)
            for i in range(nl)
        ])
        if rd > 0:
            hd = dm // nh
            for block in self.b:
                block.a.rd = rd
                block.a.ro = RY(hd, base=rb, tsl=1024, rd=rd)
        if xsn > 0:
            for i in range(max(0, nl - xsn), nl):
                self.b[i].a.ux = True
        kv_dim = nkh * (dm // nh)
        self.vli = [int(x) for x in vel.split(",") if x.strip()] if vee else []
        if self.vli:
            self.v = VE(vs, ved, kv_dim)
            self.y = PL([P(ON(1, dtype=F32)) for _ in self.vli])
        else:
            self.v = None; self.y = PL()
        self.n = RN()
        self.vre = nl > 1
        if self.vre:
            self.r = PL([
                P(TT(0.0, dtype=F32)) for _ in range(nl - 1)
            ])
        else:
            self.r = PL()
        self.h = None if te else CL(dm, vs, bias=False)
        if self.h is not None: self.h._zero_init = True
        self.iw()
    def state_dict(self,*a,**k):
        d=super().state_dict(*a,**k)
        return {n[:-7]+WT if n.endswith(".weight") else n:t for n,t in d.items()}
    def load_state_dict(self,sd,*a,**k):
        sd={n[:-2]+".weight" if n.endswith(WT) or n.endswith(".z") else n:t for n,t in sd.items()}
        return super().load_state_dict(sd,*a,**k)
    def iw(self):
        if self.t:
            NI.normal_(self.x.weight, mean=0.0, std=self.ts)
        nl = len(self.b)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False): NI.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    NI.orthogonal_(module.weight, gain=1.0)
                    if ".p." in name or name.endswith(".p"):
                        with NG(): module.weight.mul_(1.0 / math.sqrt(2 * nl))
        for i, block in enumerate(self.b):
            with NG():
                phase = SG(TT(3.0 * (i / max(nl-1, 1) - 0.5)))
                block.r.data[0] = phase * ON(block.r.shape[1])
                block.r.data[1] = (1-phase) * ON(block.r.shape[1])
    def gv(self, li, ids, vc):
        if self.v is None or li not in self.vli: return None
        if 've' not in vc: vc['ve'] = self.v(ids)
        ve_idx = self.vli.index(li)
        return vc['ve'] * TY(self.y[ve_idx], vc['ve'].dtype)
    def rl(self, x, x0, ids):
        skips, bo, xb = [], self.nl // 2, None
        vc = {}
        v0 = None
        if self.vre:
            blk0 = self.b[0]
            mix0 = TY(blk0.r, x0.dtype)
            x_in0 = mix0[0][None, None, :] * x0 + mix0[1][None, None, :] * x0
            v0 = blk0.a.v(blk0.an(x_in0) * blk0.lsf)
        vi = 0
        for i in range(self.nel):
            ve = self.gv(i, ids, vc)
            v_res = None
            if i > 0 and v0 is not None:
                alpha = SG(TY(self.r[vi], x.dtype))
                v_res = alpha * v0
                vi += 1
            x = self.b[i](x, x0, ve=ve, vr=v_res); skips.append(x)
            if i == bo: xb = x
        for i in range(self.ndl):
            li = self.nel + i
            if skips: x = x + TY(self.w[i], x.dtype)[None, None, :] * skips.pop()
            ve = self.gv(li, ids, vc)
            v_res = None
            if v0 is not None:
                alpha = SG(TY(self.r[vi], x.dtype))
                v_res = alpha * v0
                vi += 1
            x = self.b[li](x, x0, ve=ve, vr=v_res)
            if li == bo and xb is None: xb = x
        if self.o is not None and xb is not None:
            x = x - self.o.to(x.dtype) * xb
        return x
    def eb(self, ids):
        x = self.x(ids)
        if self.g is not None: x = x + self.g(ids)
        x = RM(x, (self.x.weight.shape[1],))
        if self.s is not None: x = self.s(x)
        return x
    def forward(self, ids, tgt):
        x0 = self.eb(ids); x = self.rl(x0, x0, ids)
        x_flat = RS(self.n(x), -1, x.size(-1)); targets = RS(tgt, -1)
        logits_proj = LI(x_flat, self.x.weight) if self.t else self.h(x_flat)
        logits = self.l * th.tanh(logits_proj / self.l)
        return F.cross_entropy(TF(logits), targets, reduction="mean")
    def fwl(self, ids):
        x0 = self.eb(ids); x = self.n(self.rl(x0, x0, ids))
        logits = LI(x, self.x.weight.to(x.dtype)) if self.t else self.h(x)
        return self.l * th.tanh(logits / self.l)

def ch(bm, tl, args, dv, gas, nbc=256):
    hh = {}
    hooks = []
    for n, m in bm.named_modules():
        if isinstance(m, CL):
            pn = n + WT
            c = m.weight.shape[1]
            hh[pn] = Z(c, c, dtype=F32, device='cpu')
            def mh(pn):
                ct = [0]
                def hf(_, inp, __):
                    x = TF(inp[0].detach())
                    if x.ndim == 3:
                        x = RS(x, -1, x.shape[-1])
                    xtx = (x.T @ x).cpu()
                    hh[pn] += xtx
                    ct[0] += x.shape[0]
                return hf
            h = m.register_forward_hook(mh(pn))
            hooks.append(h)
    bm.eval()
    with IM(), AU():
        for _ in range(nbc):
            x, y = tl.nb(args.tbt, args.tsl, gas)
            _ = bm(x, y)
    for h in hooks: h.remove()
    for name in hh:
        H = hh[name]
        H /= nbc
        damp = 0.01 * DG(H).mean().clamp_min(1e-6)
        H += damp * EYE(H.shape[0])
        hh[name] = H
    bm.train()
    return hh

def vl(lfn, rk, ws, dv, vt,
                     bb, hs, ib,
                     ql, stride, ebs=256):
    tt = vt.numel() - 1; ww, p = [], 0
    while p + ql <= tt:
        s = 0 if p == 0 else (ql - stride); ww.append((p, s)); p += stride
    n = len(ww); pr = (n + ws - 1) // ws
    mw = ww[rk*pr:min((rk+1)*pr, n)]
    ls = Z((), device=dv, dtype=F64)
    tc = Z((), device=dv, dtype=F64)
    bc = Z((), device=dv, dtype=F64)
    with IM():
        for i in range(0, len(mw), ebs):
            bt = mw[i:i+ebs]; bs = len(bt)
            xl = [vt[w:w+ql] for w, _ in bt]
            yl = [vt[w+1:w+ql+1] for w, _ in bt]
            pad = ebs - bs
            if pad > 0: xl.extend([xl[-1]]*pad); yl.extend([yl[-1]]*pad)
            x = SK(xl).to(device=dv, dtype=I64)
            y = SK(yl).to(device=dv, dtype=I64)
            with AU(): logits = lfn(x)
            for b in range(bs):
                s = bt[b][1]; sl, st = logits[b, s:], y[b, s:]
                ls += F.cross_entropy(TF(sl), st, reduction="sum").to(F64)
                ns = st.numel(); tc += ns
                prev, tgt = x[b, s:s+ns], st
                tb = bb[tgt].to(I16)
                tb += (hs[tgt] & ~ib[prev]).to(I16)
                bc += tb.to(F64).sum()
    if IA() and II():
        for t in [ls, tc, bc]: ARD(t, op=ROP.SUM)
    vl = IT(ls / tc)
    return vl, vl / math.log(2.0) * (IT(tc) / IT(bc))

def rp(sp: str) -> str:
    if not sp:
        return ""
    if sp == "1":
        return str(PT(EG("OUT_DIR", ".")) / "pre_export_model.pt")
    return sp

def mp(ps: str, sd: dict[str, th.Tensor]):
    cp = rp(ps)
    if not cp:
        return ""
    ckpt = {MS: {k: DC(v) for k, v in sd.items()}}
    path = PT(cp)
    path.parent.mkdir(parents=True, exist_ok=True)
    tp = path.with_suffix(path.suffix + ".tmp")
    th.save(ckpt, tp)
    os.replace(tp, path)
    return str(path)

def re(a, bm, rk, ws, dv, dd, m0,
                    code, vt, bb, hs, ib, log0):
    cl = DTL(a.tf, rk, ws, dv)
    hh = ch(bm, cl, a, dv, 8 // ws,
                                nbc=a.gcb)
    hm = {}
    for n, m in bm.named_modules():
        if isinstance(m, CL):
            k = n + WT
            if k in hh: hm[k] = hh[k]
    sd_cpu = {k: DC(v) for k, v in bm.state_dict().items()}
    cb = len(code.encode(U)); sl = 16_000_000
    qr, qm = qs(sd_cpu, hh=hm)
    if a.pp > 0:
        a6 = []
        for name, info in qm.items():
            if mk(info) == "6":
                qname = name + ".q"
                if qname in qr:
                    a6.append(TF(qr[qname].flatten().abs()))
        if a6:
            av = CAT(a6)
            k = max(1, int(a.pp * av.numel()))
            thr = IT(av.kthvalue(k).values)
            prc = 0
            for name, info in qm.items():
                if mk(info) == "6":
                    qname = name + ".q"
                    if qname in qr:
                        mask = qr[qname].abs() <= int(thr)
                        prc += IT(mask.sum())
                        qr[qname][mask] = 0
            ti6 = sum(qr[n + ".q"].numel() for n, i in qm.items() if mk(i) == "6" and n + ".q" in qr)
            log0(f"prune:{prc}/{ti6} ({100*prc/max(ti6,1):.1f}%) thr={thr:.0f}")
    eb, en = eq(qm)
    d16 = eb[:4] == Q1 and all(d < 65536 for t in qr.values() for d in t.shape)
    if d16: eb = QMB + eb[4:]
    rf, sf, ss, rh = (F6, F7, 2, 4) if d16 else (F5, F4, 4, 5)
    nti = {name: idx for idx, name in enumerate(en)}
    parts = [PK(F4, len(eb)), eb]
    mb0 = 4 + len(eb)
    thb = 0
    p6b = 0
    opb = 0
    p6t = 0
    for tname in sorted(qr):
        t = qr[tname]
        bn = tname[:-2] if tname.endswith(QS) else ""
        pi = (
            tname.endswith(QS)
            and mk(qm.get(bn)) == "6"
        )
        dm0 = {I8: 0, F16: 1, F32: 2, BF: 3}
        dt = 5 if pi else dm0.get(t.dtype, 2)
        if pi:
            raw = p6(t)
        else:
            t_np = CG(t).numpy() if t.dtype != BF else CG(t).view(U16).numpy()
            raw = t_np.tobytes()
        ni, suffix = tr(tname, nti)
        parts.append(PK(rf, ni, suffix, dt, t.ndim))
        thb += rh + ss * t.ndim
        for d in t.shape: parts.append(PK(sf, d))
        parts.append(raw)
        if pi:
            p6b += len(raw)
            p6t += 1
        else:
            opb += len(raw)
    quant_raw = b"".join(parts)
    model_blob, mc = cm(quant_raw)
    mb = len(model_blob); ts = cb + mb
    log0(
        "ab:"
        f" m={mb0}"
        f" th={thb}"
        f" p6={p6b}"
        f" op={opb}"
        f" rt={len(quant_raw)}"
        f" cm={mb}"
        f" cd={cn(mc)}"
        f" t6={p6t}"
    )
    log0(f"sz:m={mb} c={cb} t={ts}({ts/1e6:.2f}M)")
    if ts > sl: log0(f"warn:size {ts}+{ts - sl}")
    else: log0(f"size_ok:{ts/1e6:.2f} MB")
    if m0:
        with open(FM, "wb") as f: f.write(model_blob)
    if dd: BR()
    with open(FM, "rb") as f: md = f.read()
    rd, _ = db(md)
    o = 0
    ml = UF(F4, rd, o)[0]; o += 4
    lm, en, ctr = dq(rd[o:o+ml]); o += ml
    drm = {0: (I8, np.int8), 1: (F16, np.float16), 2: (F32, np.float32), 3: (BF, np.uint16)}
    lr = {}
    while o < len(rd):
        if ctr:
            ni, suffix, dt, ndim = UF(F6 if ctr == 3 else F5, rd, o); o += 4 + (ctr < 3)
            tname = rt(ni, suffix, en)
            sf, ss = (F7, 2) if ctr == 3 else (F4, 4)
        else:
            nl = UF("<H", rd, o)[0]; o += 2
            tname = rd[o:o+nl].decode(U); o += nl
            dt, ndim = UF("<BB", rd, o); o += 2
            sf, ss = F4, 4
        shape = []
        for _ in range(ndim):
            shape.append(UF(sf, rd, o)[0]); o += ss
        if dt == 4:
            numel = NM(shape)
            nbytes = ((numel + 3) // 4) * 3
            raw = memoryview(rd)[o:o+nbytes]
            o += nbytes
            t = u6l(raw, shape)
        elif dt == 5:
            numel = NM(shape)
            nbytes = ((numel + 7) // 8) * 6
            raw = memoryview(rd)[o:o+nbytes]
            o += nbytes
            t = u6(raw, shape)
        else:
            td, nd = drm[dt]
            numel = NM(shape)
            nbytes = numel * np.dtype(nd).itemsize
            arr = FB(rd, dtype=nd, count=numel, offset=o).copy()
            o += nbytes
            t = RS(FN(arr), shape)
            if td == BF: t = t.view(BF)
        lr[tname] = t
    deq_state = ds(lr, lm, sd_cpu)
    bm.load_state_dict(deq_state, 1)
    eval_sl = a.esl if a.esl > 0 else a.tsl
    vte = lv(a.val_files, eval_sl) if eval_sl != a.tsl else vt
    rlf = CMP(bm.fwl, dynamic=False) if not GB(TD) else bm.fwl
    wx = Z(a.ebs, eval_sl, dtype=I64, device=dv)
    bm.eval()
    with IM(), AU(): _ = rlf(wx)
    SY(); te = PC()
    q_vl, q_vb = vl(rlf, rk, ws, dv,
        vte, bb, hs, ib,
        eval_sl, a.evs, ebs=a.ebs)
    SY(); et = PC() - te
    log0(f"final_int6 val_loss:{q_vl:.4f} val_bpb:{q_vb:.4f} eval_time:{et*1000:.0f}ms")
    log0(f"final_int6_exact val_loss:{q_vl:.8f} val_bpb:{q_vb:.8f}")
    if dd: DGP()

def main():
    global z5
    code = PT(__file__).read_text(encoding=U); a = H()
    z5 = CMP(z5)
    dd = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rk = GI("RANK"); ws = GI("WORLD_SIZE", 1)
    lrk = GI("LOCAL_RANK")
    if ws <= 0 or 8 % ws != 0: raise ER(f"bad WS={ws}")
    gas = 8 // ws; grad_scale = 1.0 / gas
    if not th.cuda.is_available(): raise RE("cuda req")
    dv = th.device(CU, lrk); th.cuda.set_device(dv)
    if dd: IGP(backend="nccl", device_id=dv); BR()
    m0 = rk == 0
    th.backends.cuda.matmul.allow_tf32 = True; th.backends.cudnn.allow_tf32 = True
    if not HAS_FA3:
        from th.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
        enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)
    lf = None
    if m0: os.makedirs("logs", exist_ok=True); lf = f"logs/{a.rid}.txt"; print(lf)
    def log0(msg):
        if not m0: return
        print(msg)
        if lf:
            with open(lf, "a", encoding=U) as f: print(msg, file=f)
    random.seed(a.sd); np.random.seed(a.sd); th.manual_seed(a.sd); th.cuda.manual_seed_all(a.sd)
    sp = spm.SentencePieceProcessor(model_file=a.tp)
    vt = lv(a.vf, a.tsl)
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
        if isinstance(m, CL): TF(m)
    rf32(bm)
    gs = bm.state_dict; ls = bm.load_state_dict
    cm = CMP(bm, dynamic=False, fullgraph=True) if not GB(TD) else bm
    model = DDP(cm, device_ids=[lrk], broadcast_buffers=False) if dd else cm
    eoc = rp(a.eoc)
    if eoc:
        ckpt = th.load(eoc, map_location="cpu")
        ms = ckpt[MS] if isinstance(ckpt, dict) and MS in ckpt else ckpt
        ls(ms, 1)
        re(a, bm, rk, ws, dv, dd, m0,
                        code, vt, bb, hs, ib, log0)
        return
    bnp = list(bm.b.named_parameters())
    mpa = [p for n, p in bnp if p.ndim == 2 and not IC(n)]
    spa = [p for n, p in bnp if p.ndim < 2 or IC(n)]
    if bm.w.numel() > 0: spa.append(bm.w)
    if bm.s is not None: spa.append(bm.s.g)
    if bm.o is not None: spa.append(bm.o)
    if bm.g is not None: spa.append(bm.g.s)
    if bm.v is not None:
        spa.append(bm.v.s)
        for s in bm.y: spa.append(s)
    if bm.vre:
        for alpha in bm.r: spa.append(alpha)
    tlr = a.telr if a.te else a.elr
    tpg = [{PA: [bm.x.weight], "lr": tlr, BL: tlr}]
    if bm.g is not None:
        tpg.append({PA: [bm.g.e.weight], "lr": tlr, BL: tlr})
        if bm.g.p is not None: mpa.append(bm.g.p.weight)
    if bm.v is not None:
        tpg.append({PA: [bm.v.e.weight], "lr": tlr, BL: tlr})
        if bm.v.p is not None: mpa.append(bm.v.p.weight)
    ot = th.optim.AdamW(tpg, betas=(a.beta1, a.beta2), eps=a.aep, weight_decay=a.awd, fused=True)
    om = MU(mpa, lr=a.mlr, momentum=a.mum, ns_steps=a.mns, wd=a.mwd)
    for group in om.param_groups: group[BL] = a.mlr
    os = th.optim.AdamW([{PA: spa, "lr": a.slr, BL: a.slr}],
                                          betas=(a.beta1, a.beta2), eps=a.aep, weight_decay=a.awd, fused=True)
    opts = [ot, om, os]
    if bm.h is not None:
        oh = th.optim.Adam([{PA: [bm.h.weight], "lr": a.hlr, BL: a.hlr}],
                                           betas=(a.beta1, a.beta2), eps=a.aep, fused=True)
        opts.insert(1, oh)
    tl = DTL(a.tf, rk, ws, dv)
    def zga():
        for opt in opts: opt.zero_grad(set_to_none=True)
    wm = 1000.0 * a.mws if a.mws > 0 else None
    def lr_mul(step, ems):
        if a.wdi <= 0: return 1.0
        if wm is None:
            wd0 = max(a.it - a.wdi, 0)
            return max((a.it - step) / max(a.wdi, 1), 0.0) if step >= wd0 else 1.0
        step_ms = ems / max(step, 1); wd_ms = a.wdi * step_ms
        rem_ms = max(wm - ems, 0.0)
        return rem_ms / max(wd_ms, 1e-9) if rem_ms <= wd_ms else 1.0

    if a.wus > 0:
        is0 = {n: DL(t) for n, t in gs().items()}
        os0 = [copy.deepcopy(opt.state_dict()) for opt in opts]
        model.train()
        for wi in range(a.wus):
            zga()
            for ms in range(gas):
                if dd: model.require_backward_grad_sync = ms == gas - 1
                x, y = tl.nb(a.tbt, a.tsl, gas)
                with AU(): wl = model(x, y)
                (wl * grad_scale).backward()
            for opt in opts: opt.step()
            zga()
        ls(is0, 1)
        for opt, state in zip(opts,os0,strict=1): opt.load_state_dict(state)
        zga()
        if dd: model.require_backward_grad_sync = True
        tl = DTL(a.tf, rk, ws, dv)
    es = {name: TF(t.detach()).clone() for name, t in gs().items()}
    tm, ss = 0.0, None
    sws, swc = None, 0
    SY(); t0 = PC(); step = 0
    while True:
        last_step = step == a.it or (ss is not None and step >= ss)
        sv = last_step or (a.vle > 0 and step % a.vle == 0)
        if sv:
            SY(); tm += 1000.0 * (PC() - t0)
            vl, vb = vv(a, model, rk, ws, dv, gas,
                              vt, bb, hs, ib)
            log0(f"step:{step}/{a.it} val_loss:{vl:.4f} val_bpb:{vb:.4f} train_time:{tm:.0f}ms step_avg:{tm/max(step,1):.2f}ms")
            SY(); t0 = PC()
        if last_step:
            if ss is not None and step < a.it:
                log0(f"stop:wall train:{tm:.0f}ms step:{step}/{a.it}")
            break
        ems = tm + 1000.0 * (PC() - t0)
        scale = lr_mul(step, ems)
        if a.lqt > 0 and scale < a.lqt and not CL._qat_enabled:
            CL._qat_enabled = True
        zga(); tls = Z((), device=dv)
        for ms in range(gas):
            if dd: model.require_backward_grad_sync = ms == gas - 1
            x, y = tl.nb(a.tbt, a.tsl, gas)
            with AU(): loss = model(x, y)
            tls += loss.detach(); (loss * grad_scale).backward()
        tls /= gas
        frac = min(step / a.mmws, 1.0) if a.mmws > 0 else 1.0
        for group in om.param_groups:
            group["momentum"] = (1-frac)*a.mmst + frac*a.mum
        for opt in opts:
            for group in opt.param_groups: group["lr"] = group[BL] * scale
        if a.gcn > 0: th.nn.utils.clip_grad_norm_(bm.parameters(), a.gcn)
        for opt in opts: opt.step()
        zga()
        with NG():
            for name, t in gs().items():
                es[name].mul_(a.ed).add_(TF(t.detach()), alpha=1.0 - a.ed)
        step += 1
        cms = tm + 1000.0 * (PC() - t0)
        if a.swe and scale < 0.2 and step % a.swi == 0:
            if sws is None:
                sws = {n: DL(t) for n, t in gs().items()}
                swc = 1
            else:
                for n, t in gs().items(): sws[n] += DC(t)
                swc += 1
        if a.tle > 0 and (step <= 10 or step % a.tle == 0):
            log0(f"step:{step}/{a.it} train_loss:{IT(tls):.4f} train_time:{cms:.0f}ms step_avg:{cms/step:.2f}ms")
        rc = wm is not None and cms >= wm
        if dd and wm is not None:
            rct = TT(int(rc), device=dv); ARD(rct, op=ROP.MAX); rc = bool(IT(rct))
        if ss is None and rc: ss = step
    cs = gs()
    avs = {name: TY(t, cs[name].dtype) for name, t in es.items()}
    ls(avs, 1)
    mp(a.spc, gs())
    re(a, bm, rk, ws, dv, dd, m0,
                    code, vt, bb, hs, ib, log0)

if __name__ == "__main__":
    main()
