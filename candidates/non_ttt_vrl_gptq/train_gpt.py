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
AC=th.autocast; IM=th.inference_mode; Z=th.zeros; TT=th.tensor; QT=th.quantile; EM=th.empty; FN=th.from_numpy; SK=th.stack
P=nn.Parameter; PL=nn.ParameterList; M=nn.Module; ML=nn.ModuleList; NI=nn.init; NG=th.no_grad; CLP=th.clamp; DG=th.diag; CMP=th.compile; SY=th.cuda.synchronize
IA=dist.is_available; II=dist.is_initialized; BR=dist.barrier; GWS=dist.get_world_size; GRK=dist.get_rank; IGP=dist.init_process_group; DGP=dist.destroy_process_group; ARD=dist.all_reduce; ROP=dist.ReduceOp
ZL=th.zeros_like; EL=th.empty_like; AR=th.arange; SG=th.sigmoid; CAT=th.cat; ON=th.ones; FUL=th.full; EYE=th.eye
try:
    from flash_attn_interface import flash_attn_func as _fa3_func
    HAS_FA3 = True
except ImportError:
    HAS_FA3 = False

class H:
    dp = EG("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tf = JP(dp, "fineweb_train_*.bin")
    vf = JP(dp, "fineweb_val_*.bin")
    tp = EG("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    rid = EG("RUN_ID", str(uuid.uuid4()))
    sd = int(EG("SEED", 42))
    vbs = int(EG("VAL_BATCH_SIZE", 524_288))
    vle = int(EG("VAL_LOSS_EVERY", 500))
    tle = int(EG("TRAIN_LOG_EVERY", 200))
    it = int(EG("ITERATIONS", 20000))
    wus = int(EG("WARMUP_STEPS", 20))
    tbt = int(EG("TRAIN_BATCH_TOKENS", 786_432))
    tsl = int(EG("TRAIN_SEQ_LEN", 2048))
    esl = int(EG("EVAL_SEQ_LEN", 2048))
    evs = int(EG("EVAL_STRIDE", 64))
    ebs = int(EG("EVAL_BATCH_SEQS", 256))
    mws = float(EG("MAX_WALLCLOCK_SECONDS", 600.0))
    qgi = float(EG("QK_GAIN_INIT", 1.5))
    vs = int(EG("VOCAB_SIZE", 1024))
    nl = int(EG("NUM_LAYERS", 11))
    nkh = int(EG("NUM_KV_HEADS", 4))
    dm = int(EG("MODEL_DIM", 512))
    nh = int(EG("NUM_HEADS", 8))
    mm = int(EG("MLP_MULT", 3))
    te = bool(int(EG("TIE_EMBEDDINGS", "1")))
    rb = float(EG("ROPE_BASE", 10000.0))
    lsc = float(EG("LOGIT_SOFTCAP", 30.0))
    elr = float(EG("EMBED_LR", 0.6))
    hlr = float(EG("HEAD_LR", 0.008))
    telr = float(EG("TIED_EMBED_LR", 0.035))
    teis = float(EG("TIED_EMBED_INIT_STD", 0.005))
    mlr = float(EG("MATRIX_LR", 0.025))
    slr = float(EG("SCALAR_LR", 0.025))
    mum = float(EG("MUON_MOMENTUM", 0.99))
    mns = int(EG("MUON_NS_STEPS", 5))
    mmst = float(EG("MUON_MOMENTUM_WARMUP_START", 0.92))
    mmws = int(EG("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    mwd = float(EG("MUON_WD", 0.04))
    beta1 = float(EG("BETA1", 0.9))
    beta2 = float(EG("BETA2", 0.95))
    aep = float(EG("ADAM_EPS", 1e-8))
    awd = float(EG("ADAM_WD", 0.04))
    gcn = float(EG("GRAD_CLIP_NORM", 0.3))
    se = bool(int(EG("SMEAR_ENABLED", "1")))
    be = bool(int(EG("BACKOUT_ENABLED", "0")))
    boi = float(EG("BACKOUT_INIT", 0.2))
    ed = float(EG("EMA_DECAY", 0.997))
    swe = bool(int(EG("SWA_ENABLED", "1")))
    swi = int(EG("SWA_INTERVAL", 50))
    wdi = int(EG("WARMDOWN_ITERS", 3500))
    lqt = float(EG("LATE_QAT_THRESHOLD", 0.15))
    bgvs = int(EG("BIGRAM_VOCAB_SIZE", 2048))
    bgd = int(EG("BIGRAM_DIM", 128))
    xsn = int(EG("XSA_LAST_N", 11))
    rd = int(EG("ROPE_DIMS", 16))
    lns = bool(int(EG("LN_SCALE", "1")))
    vee = bool(int(EG("VE_ENABLED", "1")))
    ved = int(EG("VE_DIM", 128))
    vel = EG("VE_LAYERS", "9,10")
    gcb = int(EG("GPTQ_CALIB_BATCHES", 256))
    gbs = int(EG("GPTQ_BLOCK_SIZE", 128))
    qcp = float(EG("QAT_CLIP_PCT", 0.9995))
    pp = float(EG("PRUNE_PCT", 0.02))
    spc = EG("SAVE_PRE_EXPORT_CHECKPOINT", "")
    eoc = EG("EXPORT_ONLY_CHECKPOINT", "")

def z5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16(); X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed: X = X.T
    for _ in range(steps):
        A = X @ X.T; B = b * A + c * A @ A; X = a * X + B @ X
    return X.T if transposed else X

class Muon(th.optim.Optimizer):
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

def build_sentencepiece_luts(sp, vs, device):
    sp_vocab_size = int(sp.vocab_size()); table_size = max(sp_vocab_size, vs)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for tid in range(sp_vocab_size):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid): continue
        is_boundary_token_np[tid] = False
        if sp.is_byte(tid): base_bytes_np[tid] = 1; continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"): has_leading_space_np[tid] = True; piece = piece[1:]
        base_bytes_np[tid] = len(piece.encode("utf-8"))
    return (TT(base_bytes_np, dtype=I16, device=device),
            TT(has_leading_space_np, dtype=BO, device=device),
            TT(is_boundary_token_np, dtype=BO, device=device))

def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files: raise FileNotFoundError(f"no:{pattern}")
    tokens = CAT([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0: raise ValueError(f"val<{seq_len}")
    return tokens[:usable + 1]

def eval_val(a, model, rank, ws, device, gas,
             vt, bb, hs, ib, esl=0):
    seq_len = esl if esl > 0 else a.tsl
    lbs = a.vbs // (ws * gas) // seq_len
    total_seqs = (vt.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // ws; seq_end = (total_seqs * (rank + 1)) // ws
    val_loss_sum = Z((), device=device, dtype=F64)
    val_token_count = Z((), device=device, dtype=F64)
    val_byte_count = Z((), device=device, dtype=F64)
    model.eval()
    with IM():
        for bss in range(seq_start, seq_end, lbs):
            bse = min(bss + lbs, seq_end)
            local = vt[bss*seq_len:(bse*seq_len)+1].to(device=device, dtype=I64, non_blocking=True)
            x, y = local[:-1].reshape(-1, seq_len), local[1:].reshape(-1, seq_len)
            with AC(device_type="cuda", dtype=BF, enabled=True):
                batch_loss = model(x, y).detach()
            val_loss_sum += batch_loss.to(F64) * float(y.numel())
            val_token_count += float(y.numel())
            tb = bb[y.reshape(-1)].to(dtype=I16)
            tb += (hs[y.reshape(-1)] & ~ib[x.reshape(-1)]).to(dtype=I16)
            val_byte_count += tb.to(F64).sum()
    if IA() and II():
        for t in [val_loss_sum, val_token_count, val_byte_count]: ARD(t, op=ROP.SUM)
    val_loss = val_loss_sum / val_token_count
    bpt = val_loss.item() / math.log(2.0); tpb = val_token_count.item() / val_byte_count.item()
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

def _classify_param(name):
    if "tok_emb" in name or "lm_head" in name: return "embed"
    if ".mlp." in name: return "mlp"
    if "bigram" in name: return "bigram"
    if ".attn." in name or (".proj." in name and ".mlp." not in name): return "attn"
    if "ve_shared" in name: return "ve"
    return "other"

def _meta_kind(info):
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
    parts = [QMB, struct.pack("<H", len(names))]
    for name in names:
        kind = meta[name]
        kind_code = kind_map.get(kind)
        if kind_code is None:
            kind_name = _meta_kind(kind)
            if kind_name == "int6": kind_code = 2
            elif kind_name == "int8": kind_code = 3
            else: raise ValueError(f"bad meta {name}:{kind!r}")
        name_bytes = name.encode("utf-8")
        parts.append(struct.pack("<HB", len(name_bytes), kind_code))
        parts.append(name_bytes)
    return b"".join(parts), names

def dqm(blob: bytes) -> tuple[dict[str, object], list[str], bool]:
    if blob.startswith(QMB):
        offset = len(QMB)
        entry_count = struct.unpack_from("<H", blob, offset)[0]; offset += 2
        kind_map = {
            0: MP,
            1: MC,
            2: M6,
            3: M8,
        }
        meta, names = {}, []
        for _ in range(entry_count):
            name_len, kind_code = struct.unpack_from("<HB", blob, offset); offset += 3
            name = blob[offset:offset+name_len].decode("utf-8"); offset += name_len
            meta[name] = kind_map[kind_code]
            names.append(name)
        return meta, names, True
    return json.loads(blob.decode("utf-8")), [], False

def etr(tname: str, name_to_idx: dict[str, int]) -> tuple[int, int]:
    if tname in name_to_idx:
        return name_to_idx[tname], 0
    if tname.endswith(".q") and tname[:-2] in name_to_idx:
        base_name, suffix = tname[:-2], 1
    elif tname.endswith(".scale") and tname[:-6] in name_to_idx:
        base_name, suffix = tname[:-6], 2
    else:
        base_name, suffix = tname, 0
    return name_to_idx[base_name], suffix

def dtr(name_idx: int, suffix: int, names: list[str]) -> str:
    base_name = names[name_idx]
    if suffix == 1: return base_name + ".q"
    if suffix == 2: return base_name + ".scale"
    return base_name

def quantize_int6_gptq(weight, hessian=None, clip_range=31, block_size=128):
    t32 = weight.float()
    if t32.ndim != 2 or hessian is None:
        return _quantize_int6_percentile(t32, clip_range)
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
        return _quantize_int6_percentile(t32, clip_range)
    best_q = None; best_scale = None; best_err = float('inf')
    for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
        if pct < 1.0:
            row_clip = QT(t32.abs(), pct, dim=1)
        else:
            row_clip = t32.abs().amax(dim=1)
        s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(F16)
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
                q = CLP(th.round(w / sf), -clip_range, clip_range).to(I8)
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

def _quantize_int6_percentile(t32, clip_range=31):
    if t32.ndim == 2:
        best_q, best_s, best_err = None, None, float('inf')
        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
            if pct < 1.0:
                row_clip = QT(t32.abs(), pct, dim=1)
            else:
                row_clip = t32.abs().amax(dim=1)
            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(F16)
            q = CLP(th.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(I8)
            recon = q.float() * s.float()[:, None]
            err = (t32 - recon).pow(2).mean().item()
            if err < best_err:
                best_q, best_s, best_err = q, s, err
        return best_q, best_s
    amax = t32.abs().max().item()
    scale = TT(amax / clip_range if amax > 0 else 1.0, dtype=F16)
    q = CLP(th.round(t32 / scale.float()), -clip_range, clip_range).to(I8)
    return q, scale

def quantize_float_tensor(t):
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

def qsd(sd, hessians=None):
    result, meta = {}, {}
    int6_cats = {"mlp", "attn", "bigram", "ve"}
    for name, tensor in sd.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        if not t.is_floating_point() or t.numel() <= I8K:
            result[name] = t.to(F16) if t.is_floating_point() else t
            meta[name] = MP; continue
        if any(p in name for p in CP):
            result[name] = t.float(); meta[name] = MC; continue
        if cat in int6_cats and t.ndim >= 1:
            H = hessians.get(name) if hessians else None
            q, s = quantize_int6_gptq(t, hessian=H)
            result[name + ".q"] = q; result[name + ".scale"] = s
            meta[name] = M6; continue
        q, s = quantize_float_tensor(t)
        result[name + ".q"] = q; result[name + ".scale"] = s
        meta[name] = M8
    return result, meta

def dsd(result, meta, template_sd):
    out = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None: continue
        orig_dtype = orig.dtype
        if _meta_kind(info) in {"passthrough", "passthrough_ctrl"}:
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
    arr = arr_i16.astype(np.int16, copy=False)
    if arr.size == 0:
        return np.empty((0,), dtype=np.uint8)
    if arr.min() < -31 or arr.max() > 31:
        raise ValueError("int6 range")
    out = np.where(arr >= 0, arr * 2, (-arr) * 2 - 1)
    return out.astype(np.uint8, copy=False)

def _zigzag_decode_int6(u8: np.ndarray) -> np.ndarray:
    u = u8.astype(np.int16, copy=False)
    return np.where((u & 1) == 0, u // 2, -((u + 1) // 2)).astype(np.int16, copy=False)

def pi6l(t: Tensor) -> bytes:
    q = t.detach().cpu().contiguous()
    if q.dtype != I8:
        raise TypeError(f"need int8, got {q.dtype}")
    arr = q.numpy().reshape(-1).astype(np.int16, copy=False)
    if arr.size == 0:
        return b""
    if arr.min() < -31 or arr.max() > 31:
        raise ValueError("int6 range")
    u = (arr + 31).astype(np.uint8, copy=False)
    pad = (-u.size) % 4
    if pad:
        u = np.pad(u, (0, pad), constant_values=0)
    groups = u.reshape(-1, 4).astype(np.uint32, copy=False)
    packed = (groups[:, 0]
              | (groups[:, 1] << 6)
              | (groups[:, 2] << 12)
              | (groups[:, 3] << 18))
    out = np.empty(packed.size * 3, dtype=np.uint8)
    out[0::3] = (packed & 0xFF).astype(np.uint8)
    out[1::3] = ((packed >> 8) & 0xFF).astype(np.uint8)
    out[2::3] = ((packed >> 16) & 0xFF).astype(np.uint8)
    return out.tobytes()

def ui6l(raw: bytes | memoryview, shape: list[int]) -> Tensor:
    numel = int(np.prod(shape, dtype=np.int64))
    if numel == 0:
        return EM(shape, dtype=I8)
    packed_u8 = np.frombuffer(raw, dtype=np.uint8)
    groups = (numel + 3) // 4
    if packed_u8.size != groups * 3:
        raise ValueError(f"bad int6 sz {groups*3}!={packed_u8.size}")
    triplets = packed_u8.reshape(-1, 3).astype(np.uint32, copy=False)
    packed = triplets[:, 0] | (triplets[:, 1] << 8) | (triplets[:, 2] << 16)
    u = np.empty(groups * 4, dtype=np.uint8)
    u[0::4] = (packed & 0x3F).astype(np.uint8)
    u[1::4] = ((packed >> 6) & 0x3F).astype(np.uint8)
    u[2::4] = ((packed >> 12) & 0x3F).astype(np.uint8)
    u[3::4] = ((packed >> 18) & 0x3F).astype(np.uint8)
    arr = u[:numel].astype(np.int16, copy=False) - 31
    return FN(arr.astype(np.int8, copy=False).reshape(shape))

def pi6(t: Tensor) -> bytes:
    q = t.detach().cpu().contiguous()
    if q.dtype != I8:
        raise TypeError(f"need int8, got {q.dtype}")
    arr = q.numpy().reshape(-1).astype(np.int16, copy=False)
    if arr.size == 0:
        return b""
    u = _zigzag_encode_int6(arr.astype(np.int16, copy=False))
    numel = u.size
    pad = (-numel) % 8
    if pad:
        u = np.pad(u, (0, pad), constant_values=0)
    out = bytearray()
    for bit in range(6):
        bits = ((u >> bit) & 1).astype(np.uint8, copy=False).reshape(-1, 8)
        out.extend(np.packbits(bits, axis=1, bitorder="little").reshape(-1).tobytes())
    return bytes(out)

def ui6(raw: bytes | memoryview, shape: list[int]) -> Tensor:
    numel = int(np.prod(shape, dtype=np.int64))
    if numel == 0:
        return EM(shape, dtype=I8)
    padded = ((numel + 7) // 8) * 8
    plane_bytes = padded // 8
    packed_u8 = np.frombuffer(raw, dtype=np.uint8)
    expected = plane_bytes * 6
    if packed_u8.size != expected:
        raise ValueError(f"bad int6 sz {expected}!={packed_u8.size}")
    u = np.zeros(padded, dtype=np.uint8)
    offset = 0
    for bit in range(6):
        plane = packed_u8[offset:offset + plane_bytes]
        offset += plane_bytes
        bits = np.unpackbits(plane, bitorder="little")
        u |= (bits.astype(np.uint8, copy=False) << bit)
    arr = _zigzag_decode_int6(u[:numel])
    return FN(arr.astype(np.int8, copy=False).reshape(shape))

def mcn(codec_id: int) -> str:
    if codec_id == KZ:
        return "zstd19"
    if codec_id == KG:
        return "zlib9"
    if codec_id == KL:
        return "lzma_raw_hc3_16mb"
    return f"unknown({codec_id})"

def cmb(raw: bytes) -> tuple[bytes, int]:
    candidates = [
        (KL, lzma.compress(raw, format=lzma.FORMAT_RAW, filters=LF)),
        (KG, zlib.compress(raw, level=9)),
    ]
    if HAS_ZSTD:
        candidates.append((KZ, zstd.ZstdCompressor(level=19).compress(raw)))
    codec_id, payload = min(candidates, key=lambda item: len(item[1]))
    return QCB + bytes([codec_id]) + payload, codec_id

def dmb(blob: bytes) -> tuple[bytes, str]:
    if blob.startswith(QCB):
        codec_id = blob[len(QCB)]
        payload = blob[len(QCB) + 1:]
        if codec_id == KZ:
            if not HAS_ZSTD:
                raise RuntimeError("zstd unavailable")
            return zstd.ZstdDecompressor().decompress(payload), mcn(codec_id)
        if codec_id == KG:
            return zlib.decompress(payload), mcn(codec_id)
        if codec_id == KL:
            try:
                return lzma.decompress(payload, format=lzma.FORMAT_RAW, filters=LF), mcn(codec_id)
            except lzma.LZMAError:
                return lzma.decompress(payload), "legacy_lzma_hc4_32mb_xz"
        raise ValueError(f"bad codec {codec_id}")
    if HAS_ZSTD:
        try:
            return zstd.ZstdDecompressor().decompress(blob), "legacy_zstd22"
        except Exception:
            pass
    return zlib.decompress(blob), "legacy_zlib9"

def load_data_shard(file):
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1: raise ValueError(f"bad hdr {file}")
    num_tokens = int(header[2])
    if file.stat().st_size != header_bytes + num_tokens * np.dtype("<u2").itemsize: raise ValueError(f"bad size {file}")
    return FN(np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes).astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files: raise FileNotFoundError(f"no:{pattern}")
        self.file_idx = 0; self.tokens = load_data_shard(self.files[0]); self.pos = 0
    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files); self.tokens = load_data_shard(self.files[self.file_idx]); self.pos = 0
    def take(self, n):
        chunks, remaining = [], n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0: self._advance_file(); continue
            k = min(remaining, avail); chunks.append(self.tokens[self.pos:self.pos+k]); self.pos += k; remaining -= k
        return chunks[0] if len(chunks) == 1 else CAT(chunks)

class DTL:
    def __init__(self, pattern, rank, world_size, device):
        self.rank, self.world_size, self.device = rank, world_size, device; self.stream = TokenStream(pattern)
    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        per_rank_span = global_tokens // (self.world_size * grad_accum_steps) + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span; local = chunk[start:start+per_rank_span].to(dtype=I64)
        x, y = local[:-1].reshape(-1, seq_len), local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

class RMSNorm(M):
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

def restore_low_dim_params_to_fp32(module):
    with NG():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CP)) and param.dtype != F32:
                param.data = param.data.float()

class Rotary(M):
    def __init__(self, dim, base=10000.0, tsl=1024, rd=0):
        super().__init__()
        self.dim = dim; self.base = base; self.tsl = tsl
        self.rope_dims = rd if rd > 0 else dim
        inv_freq = 1.0 / (base ** (AR(0, self.rope_dims, 2, dtype=F32) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0; self._cos_cached = self._sin_cached = None
    def forward(self, seq_len, device, dtype):
        if self._cos_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
            rd = self.rope_dims
            if seq_len > self.tsl:
                scale = seq_len / self.tsl
                new_base = self.base * (scale ** (rd / (rd - 2)))
                inv_freq = 1.0 / (new_base ** (AR(0, rd, 2, dtype=F32, device=device) / rd))
            else: inv_freq = self.inv_freq.to(device)
            freqs = th.outer(AR(seq_len, device=device, dtype=inv_freq.dtype), inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]; self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)

def apply_rotary_emb(x, cos, sin, rd=0):
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
        self.rotary = Rotary(self.head_dim, base=rb, tsl=1024)
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
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
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

class SmearGate(M):
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

class Block(M):
    def __init__(self, dim, nh, nkh, mm, rb, qgi, layer_idx=0, ln_scale=False):
        super().__init__()
        self.attn_norm, self.mlp_norm = RMSNorm(), RMSNorm()
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
        self.smear = SmearGate(dm) if se else None
        self.backout_lambda = P(backout_init * ON(1)) if be else None
        self.num_encoder_layers = nl // 2
        self.num_decoder_layers = nl - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = P(ON(self.num_skip_weights, dm, dtype=F32))
        self.blocks = ML([
            Block(dm, nh, nkh, mm, rb, qgi,
                  layer_idx=i, ln_scale=ln_scale)
            for i in range(nl)
        ])
        if rd > 0:
            head_dim = dm // nh
            for block in self.blocks:
                block.attn.rope_dims = rd
                block.attn.rotary = Rotary(head_dim, base=rb, tsl=1024, rd=rd)
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
        self.final_norm = RMSNorm()
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

def chs(bm, tl, args, device, gas, num_batches=256):
    hessians = {}
    hooks = []
    param_to_name = {}
    for name, module in bm.named_modules():
        if isinstance(module, CL):
            param_name = name + ".weight"
            param_to_name[id(module)] = param_name
            cols = module.weight.shape[1]
            hessians[param_name] = Z(cols, cols, dtype=F32, device='cpu')
            def make_hook(mod_id, pname, ncols):
                count = [0]
                def hook_fn(module, input, output):
                    x = input[0].detach().float()
                    if x.ndim == 3:
                        x = x.reshape(-1, x.shape[-1])
                    xtx = (x.T @ x).cpu()
                    hessians[pname] += xtx
                    count[0] += x.shape[0]
                return hook_fn
            h = module.register_forward_hook(make_hook(id(module), param_name, cols))
            hooks.append(h)
    bm.eval()
    with IM(), AC(device_type="cuda", dtype=BF):
        for _ in range(num_batches):
            x, y = tl.next_batch(args.tbt, args.tsl, gas)
            _ = bm(x, y)
    for h in hooks: h.remove()
    for name in hessians:
        H = hessians[name]
        H /= num_batches
        damp = 0.01 * DG(H).mean().clamp_min(1e-6)
        H += damp * EYE(H.shape[0])
        hessians[name] = H
    bm.train()
    return hessians

def eval_val_sliding(logits_fn, rank, ws, device, vt,
                     bb, hs, ib,
                     seq_len, stride, ebs=256):
    total = vt.numel() - 1; windows, p = [], 0
    while p + seq_len <= total:
        s = 0 if p == 0 else (seq_len - stride); windows.append((p, s)); p += stride
    n = len(windows); per_rank = (n + ws - 1) // ws
    my_windows = windows[rank*per_rank:min((rank+1)*per_rank, n)]
    loss_sum = Z((), device=device, dtype=F64)
    tok_count = Z((), device=device, dtype=F64)
    byte_count = Z((), device=device, dtype=F64)
    with IM():
        for i in range(0, len(my_windows), ebs):
            batch = my_windows[i:i+ebs]; bs = len(batch)
            x_list = [vt[w:w+seq_len] for w, _ in batch]
            y_list = [vt[w+1:w+seq_len+1] for w, _ in batch]
            pad = ebs - bs
            if pad > 0: x_list.extend([x_list[-1]]*pad); y_list.extend([y_list[-1]]*pad)
            x = SK(x_list).to(device=device, dtype=I64)
            y = SK(y_list).to(device=device, dtype=I64)
            with AC(device_type="cuda", dtype=BF): logits = logits_fn(x)
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
    ckpt = {"model_state": {k: v.detach().cpu() for k, v in sd.items()}}
    path = Path(ckpt_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    th.save(ckpt, tmp_path)
    os.replace(tmp_path, path)
    return str(path)

def ree(a, bm, rank, ws, device, dd, master,
                    code, vt, bb, hs, ib, log0):
    cl = DTL(a.tf, rank, ws, device)
    hessians = chs(bm, cl, a, device, 8 // ws,
                                num_batches=a.gcb)
    hm = {}
    for name, module in bm.named_modules():
        if isinstance(module, CL):
            sd_name = name + ".weight"
            h_name = name + ".weight"
            if h_name in hessians:
                hm[sd_name] = hessians[h_name]
    sd_cpu = {k: v.detach().cpu() for k, v in bm.state_dict().items()}
    cb = len(code.encode("utf-8")); sl = 16_000_000
    qr, qm = qsd(sd_cpu, hessians=hm)
    if a.pp > 0:
        ai6 = []
        for name, info in qm.items():
            if _meta_kind(info) == "int6":
                qname = name + ".q"
                if qname in qr:
                    ai6.append(qr[qname].flatten().abs().float())
        if ai6:
            all_vals = CAT(ai6)
            k = max(1, int(a.pp * all_vals.numel()))
            threshold = all_vals.kthvalue(k).values.item()
            prc = 0
            for name, info in qm.items():
                if _meta_kind(info) == "int6":
                    qname = name + ".q"
                    if qname in qr:
                        mask = qr[qname].abs() <= int(threshold)
                        prc += mask.sum().item()
                        qr[qname][mask] = 0
            ti6 = sum(qr[n + ".q"].numel() for n, i in qm.items() if _meta_kind(i) == "int6" and n + ".q" in qr)
            log0(f"prune:{prc}/{ti6} ({100*prc/max(ti6,1):.1f}%) thr={threshold:.0f}")
    meta_blob, meta_names = eqm(qm)
    name_to_idx = {name: idx for idx, name in enumerate(meta_names)}
    parts = [struct.pack("<I", len(meta_blob)), meta_blob]
    meta_bytes = 4 + len(meta_blob)
    thb = 0
    p6b = 0
    opb = 0
    p6t = 0
    tensor_order = sorted(qr.keys())
    for tname in tensor_order:
        t = qr[tname]
        base_name = tname[:-2] if tname.endswith(".q") else ""
        pack_int6 = (
            tname.endswith(".q")
            and _meta_kind(qm.get(base_name)) == "int6"
        )
        dtype_map = {I8: 0, F16: 1, F32: 2, BF: 3}
        dt = 5 if pack_int6 else dtype_map.get(t.dtype, 2)
        if pack_int6:
            raw = pi6(t)
        else:
            t_np = t.contiguous().numpy() if t.dtype != BF else t.contiguous().view(U16).numpy()
            raw = t_np.tobytes()
        name_idx, suffix = etr(tname, name_to_idx)
        parts.append(struct.pack("<HBBB", name_idx, suffix, dt, t.ndim))
        thb += 5 + 4 * t.ndim
        for d in t.shape: parts.append(struct.pack("<I", d))
        parts.append(raw)
        if pack_int6:
            p6b += len(raw)
            p6t += 1
        else:
            opb += len(raw)
    quant_raw = b"".join(parts)
    model_blob, model_codec_id = cmb(quant_raw)
    mb = len(model_blob); ts = cb + mb
    log0(
        "ab:"
        f" meta={meta_bytes}"
        f" tensor_headers={thb}"
        f" int6_payload={p6b}"
        f" other_payload={opb}"
        f" raw_total={len(quant_raw)}"
        f" compressed_model={mb}"
        f" codec={mcn(model_codec_id)}"
        f" int6_tensors={p6t}"
    )
    log0(f"sz:m={mb} c={cb} t={ts}({ts/1e6:.2f}M)")
    if ts > sl: log0(f"warn:size {ts}+{ts - sl}")
    else: log0(f"size_ok:{ts/1e6:.2f} MB")
    if master:
        with open("final_model.int6.ptz", "wb") as f: f.write(model_blob)
    if dd: BR()
    with open("final_model.int6.ptz", "rb") as f: model_blob_loaded = f.read()
    rd, _ = dmb(model_blob_loaded)
    offset = 0
    meta_len = struct.unpack_from("<I", rd, offset)[0]; offset += 4
    lm, meta_names, ctr = dqm(rd[offset:offset+meta_len]); offset += meta_len
    drm = {0: (I8, np.int8), 1: (F16, np.float16), 2: (F32, np.float32), 3: (BF, np.uint16)}
    lr = {}
    while offset < len(rd):
        if ctr:
            name_idx, suffix, dt, ndim = struct.unpack_from("<HBBB", rd, offset); offset += 5
            tname = dtr(name_idx, suffix, meta_names)
        else:
            name_len = struct.unpack_from("<H", rd, offset)[0]; offset += 2
            tname = rd[offset:offset+name_len].decode("utf-8"); offset += name_len
            dt, ndim = struct.unpack_from("<BB", rd, offset); offset += 2
        shape = []
        for _ in range(ndim):
            shape.append(struct.unpack_from("<I", rd, offset)[0]); offset += 4
        if dt == 4:
            numel = int(np.prod(shape, dtype=np.int64))
            nbytes = ((numel + 3) // 4) * 3
            raw = memoryview(rd)[offset:offset+nbytes]
            offset += nbytes
            t = ui6l(raw, shape)
        elif dt == 5:
            numel = int(np.prod(shape, dtype=np.int64))
            nbytes = ((numel + 7) // 8) * 6
            raw = memoryview(rd)[offset:offset+nbytes]
            offset += nbytes
            t = ui6(raw, shape)
        else:
            torch_dt, np_dt = drm[dt]
            numel = 1
            for s in shape: numel *= s
            nbytes = numel * np.dtype(np_dt).itemsize
            arr = np.frombuffer(rd, dtype=np_dt, count=numel, offset=offset).copy()
            offset += nbytes
            t = FN(arr).reshape(shape)
            if torch_dt == BF: t = t.view(BF)
        lr[tname] = t
    deq_state = dsd(lr, lm, sd_cpu)
    bm.load_state_dict(deq_state, strict=True)
    eval_sl = a.esl if a.esl > 0 else a.tsl
    vte = load_validation_tokens(a.val_files, eval_sl) if eval_sl != a.tsl else vt
    raw_logits_fn = CMP(bm.forward_logits, dynamic=False) if not bool(int(EG("TORCH_COMPILE_DISABLE", "0"))) else bm.forward_logits
    warmup_x = Z(a.ebs, eval_sl, dtype=I64, device=device)
    bm.eval()
    with IM(), AC(device_type="cuda", dtype=BF): _ = raw_logits_fn(warmup_x)
    SY(); t_eval = PC()
    q_vl, q_vb = eval_val_sliding(raw_logits_fn, rank, ws, device,
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
    local_rank = int(EG("LOCAL_RANK", "0"))
    if ws <= 0 or 8 % ws != 0: raise ValueError(f"bad WS={ws}")
    gas = 8 // ws; grad_scale = 1.0 / gas
    if not th.cuda.is_available(): raise RuntimeError("cuda req")
    device = th.device("cuda", local_rank); th.cuda.set_device(device)
    if dd: IGP(backend="nccl", device_id=device); BR()
    master = rank == 0
    th.backends.cuda.matmul.allow_tf32 = True; th.backends.cudnn.allow_tf32 = True
    if not HAS_FA3:
        from th.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
        enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)
    logfile = None
    if master: os.makedirs("logs", exist_ok=True); logfile = f"logs/{a.rid}.txt"; print(logfile)
    def log0(msg, console=True):
        if not master: return
        if console: print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f: print(msg, file=f)
    random.seed(a.sd); np.random.seed(a.sd); th.manual_seed(a.sd); th.cuda.manual_seed_all(a.sd)
    sp = spm.SentencePieceProcessor(model_file=a.tp)
    vt = load_validation_tokens(a.vf, a.tsl)
    bb, hs, ib = build_sentencepiece_luts(sp, a.vs, device)
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
    ).to(device).bfloat16()
    for m in bm.modules():
        if isinstance(m, CL): m.float()
    restore_low_dim_params_to_fp32(bm)
    compiled_model = CMP(bm, dynamic=False, fullgraph=True) if not bool(int(EG("TORCH_COMPILE_DISABLE", "0"))) else bm
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if dd else compiled_model
    eoc = rcp(a.eoc)
    if eoc:
        ckpt = th.load(eoc, map_location="cpu")
        model_state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        bm.load_state_dict(model_state, strict=True)
        ree(a, bm, rank, ws, device, dd, master,
                        code, vt, bb, hs, ib, log0)
        return
    block_named_params = list(bm.blocks.named_parameters())
    matrix_params = [p for n, p in block_named_params if p.ndim == 2 and not any(pat in n for pat in CP)]
    scalar_params = [p for n, p in block_named_params if p.ndim < 2 or any(pat in n for pat in CP)]
    if bm.skip_weights.numel() > 0: scalar_params.append(bm.skip_weights)
    if bm.smear is not None: scalar_params.append(bm.smear.gate)
    if bm.backout_lambda is not None: scalar_params.append(bm.backout_lambda)
    if bm.bigram is not None: scalar_params.append(bm.bigram.scale)
    if bm.ve_shared is not None:
        scalar_params.append(bm.ve_shared.scale)
        for s in bm.ve_layer_scales: scalar_params.append(s)
    if bm.vrl_enabled:
        for alpha in bm.vrl_alphas: scalar_params.append(alpha)
    token_lr = a.telr if a.te else a.elr
    tok_param_groups = [{"params": [bm.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
    if bm.bigram is not None:
        tok_param_groups.append({"params": [bm.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if bm.bigram.proj is not None: matrix_params.append(bm.bigram.proj.weight)
    if bm.ve_shared is not None:
        tok_param_groups.append({"params": [bm.ve_shared.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if bm.ve_shared.proj is not None: matrix_params.append(bm.ve_shared.proj.weight)
    optimizer_tok = th.optim.AdamW(tok_param_groups, betas=(a.beta1, a.beta2), eps=a.aep, weight_decay=a.awd, fused=True)
    optimizer_muon = Muon(matrix_params, lr=a.mlr, momentum=a.mum, ns_steps=a.mns, wd=a.mwd)
    for group in optimizer_muon.param_groups: group["base_lr"] = a.mlr
    optimizer_scalar = th.optim.AdamW([{"params": scalar_params, "lr": a.slr, "base_lr": a.slr}],
                                          betas=(a.beta1, a.beta2), eps=a.aep, weight_decay=a.awd, fused=True)
    opts = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if bm.lm_head is not None:
        optimizer_head = th.optim.Adam([{"params": [bm.lm_head.weight], "lr": a.hlr, "base_lr": a.hlr}],
                                           betas=(a.beta1, a.beta2), eps=a.aep, fused=True)
        opts.insert(1, optimizer_head)
    tl = DTL(a.tf, rank, ws, device)
    def zero_grad_all():
        for opt in opts: opt.zero_grad(set_to_none=True)
    max_wallclock_ms = 1000.0 * a.mws if a.mws > 0 else None
    def lr_mul(step, elapsed_ms):
        if a.wdi <= 0: return 1.0
        if max_wallclock_ms is None:
            wd0 = max(a.it - a.wdi, 0)
            return max((a.it - step) / max(a.wdi, 1), 0.0) if step >= wd0 else 1.0
        step_ms = elapsed_ms / max(step, 1); wd_ms = a.wdi * step_ms
        rem_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return rem_ms / max(wd_ms, 1e-9) if rem_ms <= wd_ms else 1.0

    if a.wus > 0:
        initial_model_state = {n: t.detach().cpu().clone() for n, t in bm.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in opts]
        model.train()
        for wi in range(a.wus):
            zero_grad_all()
            for ms in range(gas):
                if dd: model.require_backward_grad_sync = ms == gas - 1
                x, y = tl.next_batch(a.tbt, a.tsl, gas)
                with AC(device_type="cuda", dtype=BF, enabled=True): wl = model(x, y)
                (wl * grad_scale).backward()
            for opt in opts: opt.step()
            zero_grad_all()
        bm.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(opts, initial_optimizer_states, strict=True): opt.load_state_dict(state)
        zero_grad_all()
        if dd: model.require_backward_grad_sync = True
        tl = DTL(a.tf, rank, ws, device)
    ema_state = {name: t.detach().float().clone() for name, t in bm.state_dict().items()}
    ttms, stop_after_step = 0.0, None
    swa_state, swa_count = None, 0
    SY(); t0 = PC(); step = 0
    while True:
        last_step = step == a.it or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (a.vle > 0 and step % a.vle == 0)
        if should_validate:
            SY(); ttms += 1000.0 * (PC() - t0)
            vl, vb = eval_val(a, model, rank, ws, device, gas,
                              vt, bb, hs, ib)
            log0(f"step:{step}/{a.it} val_loss:{vl:.4f} val_bpb:{vb:.4f} train_time:{ttms:.0f}ms step_avg:{ttms/max(step,1):.2f}ms")
            SY(); t0 = PC()
        if last_step:
            if stop_after_step is not None and step < a.it:
                log0(f"stop:wall train:{ttms:.0f}ms step:{step}/{a.it}")
            break
        elapsed_ms = ttms + 1000.0 * (PC() - t0)
        scale = lr_mul(step, elapsed_ms)
        if a.lqt > 0 and scale < a.lqt and not CL._qat_enabled:
            CL._qat_enabled = True
        zero_grad_all(); train_loss = Z((), device=device)
        for ms in range(gas):
            if dd: model.require_backward_grad_sync = ms == gas - 1
            x, y = tl.next_batch(a.tbt, a.tsl, gas)
            with AC(device_type="cuda", dtype=BF, enabled=True): loss = model(x, y)
            train_loss += loss.detach(); (loss * grad_scale).backward()
        train_loss /= gas
        frac = min(step / a.mmws, 1.0) if a.mmws > 0 else 1.0
        for group in optimizer_muon.param_groups:
            group["momentum"] = (1-frac)*a.mmst + frac*a.mum
        for opt in opts:
            for group in opt.param_groups: group["lr"] = group["base_lr"] * scale
        if a.gcn > 0: th.nn.utils.clip_grad_norm_(bm.parameters(), a.gcn)
        for opt in opts: opt.step()
        zero_grad_all()
        with NG():
            for name, t in bm.state_dict().items():
                ema_state[name].mul_(a.ed).add_(t.detach().float(), alpha=1.0 - a.ed)
        step += 1
        approx_ms = ttms + 1000.0 * (PC() - t0)
        if a.swe and scale < 0.2 and step % a.swi == 0:
            if swa_state is None:
                swa_state = {n: t.detach().cpu().clone() for n, t in bm.state_dict().items()}
                swa_count = 1
            else:
                for n, t in bm.state_dict().items(): swa_state[n] += t.detach().cpu()
                swa_count += 1
        if a.tle > 0 and (step <= 10 or step % a.tle == 0):
            log0(f"step:{step}/{a.it} train_loss:{train_loss.item():.4f} train_time:{approx_ms:.0f}ms step_avg:{approx_ms/step:.2f}ms")
        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
        if dd and max_wallclock_ms is not None:
            rct = TT(int(reached_cap), device=device); ARD(rct, op=ROP.MAX); reached_cap = bool(rct.item())
        if stop_after_step is None and reached_cap: stop_after_step = step
    current_state = bm.state_dict()
    avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
    bm.load_state_dict(avg_state, strict=True)
    mspc(a.spc, bm.state_dict(), log0)
    ree(a, bm, rank, ws, device, dd, master,
                    code, vt, bb, hs, ib, log0)

if __name__ == "__main__":
    main()
