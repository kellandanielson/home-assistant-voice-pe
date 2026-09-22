"""v8 pool gate (9/12): v6.2 baseline vs v8_1..3.
Adds: POS 'ok pickle' unseen voices (edge_okpos_holdout_16k), NEG 'okay ...' near-miss
unseen voices (edge_okneg_holdout_16k). Real memos = 'hey pickle' only (no ok-pickle memos yet)."""
import glob, os
import numpy as np
import scipy.io.wavfile as wavfile
from microwakeword.audio.clips import Clips
from microwakeword.inference import Model
W = "/home/kella/mww/"
MODELS = {"v6.2": W + "trained_models/wakeword_v6_2/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite"}
for n in (1, 2, 3):
    MODELS[f"v8.{n}"] = W + f"trained_models/wakeword_v8_{n}/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite"
CUT = [0.45, 0.56, 0.70, 0.85]

def clips_split(d, split="test", limit=None):
    c = Clips(input_directory=d, file_pattern="*.wav", max_clip_duration_s=None, remove_silence=False, random_split_seed=10, split_count=0.1)
    n = 0
    for a in c.audio_generator(split=split, repeat=1):
        yield a
        n += 1
        if limit and n >= limit:
            break
def files_all(d, sub=None):
    fs = sorted(glob.glob(os.path.join(d, "*.wav")))
    if sub:
        fs = [f for f in fs if sub in f]
    for f in fs:
        sr, a = wavfile.read(f)
        yield (a.astype(np.float32) / 32767.0) if a.dtype == np.int16 else a.astype(np.float32)
SETS = [("POS hey UNSEEN voices", lambda: files_all(W + "edge_pos_holdout_16k")),
        ("POS hey UNSEEN kid", lambda: files_all(W + "edge_pos_holdout_16k", "Maisie")),
        ("POS ok UNSEEN voices", lambda: files_all(W + "edge_okpos_holdout_16k")),
        ("POS ok UNSEEN kid", lambda: files_all(W + "edge_okpos_holdout_16k", "Maisie")),
        ("NEG okay-nearmiss UNSEEN", lambda: files_all(W + "edge_okneg_holdout_16k")),
        ("NEG edge UNSEEN voices", lambda: files_all(W + "edge_neg_holdout_16k")),
        ("NEG MUSIC vocals (test)", lambda: clips_split(W + "music_vocals_16k", "test")),
        ("NEG TV/AudioSet (test)", lambda: clips_split(W + "audioset_16k", "test")),
        ("NEG real female (test)", lambda: clips_split(W + "female_real_16k", "test"))]
res = {}
for name, path in MODELS.items():
    if not os.path.exists(path):
        print(name, "missing")
        continue
    m = Model(path)
    for label, gen in SETS:
        mx = []
        for a in gen():
            p = np.asarray(m.predict_clip(a), dtype=np.float32).ravel()
            mx.append(float(p.max()) if len(p) else 0.0)
            if hasattr(m, "reset"):
                m.reset()
        res[(name, label)] = np.array(mx)
print("%-26s %-5s %5s  %s" % ("set", "mdl", "n", "  ".join(f"@{c}" for c in CUT)))
for label, _ in SETS:
    for name in MODELS:
        mx = res.get((name, label))
        if mx is None or not len(mx):
            continue
        print("%-26s %-5s %5d  %s" % (label, name, len(mx), "  ".join(f"{100*(mx>=c).mean():5.1f}%" for c in CUT)))
    print()
print("=== REAL memos, 'hey pickle' (per-utterance wakes) ===")
def segments(a, frame=160, thr_ratio=0.10, min_gap=22, min_len=25, pad=15):
    n = len(a) // frame
    rms = np.array([np.sqrt(np.mean(a[i*frame:(i+1)*frame] ** 2)) for i in range(n)])
    thr = max(rms.max() * thr_ratio, 0.004)
    act = rms > thr
    segs = []
    st = None
    gap = 0
    for i, on in enumerate(act):
        if on:
            if st is None:
                st = i
            gap = 0
        elif st is not None:
            gap += 1
            if gap >= min_gap:
                e = i - gap
                if e - st >= min_len:
                    segs.append((st, e))
                st = None
                gap = 0
    if st is not None and n - st >= min_len:
        segs.append((st, n))
    return [a[max(0, (s-pad)*frame):min(len(a), (e+pad)*frame)] for s, e in segs]
for name, path in MODELS.items():
    if not os.path.exists(path):
        continue
    m = Model(path)
    out = []
    for who in ("Kellan", "Ann", "Peggy"):
        sr, a = wavfile.read(W + f"real_clips/{who}.wav")
        a = (a.astype(np.float32) / 32767.0) if a.dtype == np.int16 else a.astype(np.float32)
        mx = []
        for sgm in segments(a):
            p = np.asarray(m.predict_clip(sgm), dtype=np.float32).ravel()
            mx.append(float(p.max()) if len(p) else 0.0)
            if hasattr(m, "reset"):
                m.reset()
        out.append(f"{who} " + " ".join(f"@{c}:{sum(x>=c for x in mx)}/{len(mx)}" for c in CUT))
    print(f"{name:5s} " + " | ".join(out))
print("EVAL-V8-DONE")
