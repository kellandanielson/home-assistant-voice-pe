"""v9 gate: real sets = held-out every-4th utterances (val) vs trained utterances, from the 9/13 recordings only.
(8/25 memos retired 9/13.) Lab negatives as in eval_v8."""
import glob, os
import numpy as np
import scipy.io.wavfile as wavfile
from microwakeword.audio.clips import Clips
from microwakeword.inference import Model
W = "/home/kella/mww/"
def P(n): return W + f"trained_models/wakeword_{n}/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite"
MODELS = {"v6.2": P("v6_2"), "ft62a": P("v9_ft62a"), "ft62b": P("v9_ft62b"), "ft56a": P("v9_ft56a"), "s1": P("v9_s1"), "f1": P("v9_f1"), "f2": P("v9_f2")}
CUT = [0.70, 0.85, 0.90, 0.95, 0.98]
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
SETS = [("REAL Kellan (val = every 4th)", lambda: files_all(W + "real_clips_val", "Kellan")),
        ("REAL Ann (val)", lambda: files_all(W + "real_clips_val", "Ann")),
        ("REAL Peggy (val)", lambda: files_all(W + "real_clips_val", "Peggy")),
        ("REAL Leo (val)", lambda: files_all(W + "real_clips_val", "Leo")),
        ("REAL Kellan (train)", lambda: files_all(W + "real_clips_train", "Kellan")),
        ("REAL Ann (train)", lambda: files_all(W + "real_clips_train", "Ann")),
        ("REAL Peggy (train)", lambda: files_all(W + "real_clips_train", "Peggy")),
        ("REAL Leo (train)", lambda: files_all(W + "real_clips_train", "Leo")),
        ("POS hey UNSEEN tts voices", lambda: files_all(W + "edge_pos_holdout_16k")),
        ("POS ok UNSEEN tts voices", lambda: files_all(W + "edge_okpos_holdout_16k")),
        ("NEG okay-nearmiss UNSEEN", lambda: files_all(W + "edge_okneg_holdout_16k")),
        ("NEG edge UNSEEN voices", lambda: files_all(W + "edge_neg_holdout_16k")),
        ("NEG MUSIC vocals (test)", lambda: clips_split(W + "music_vocals_16k", "test")),
        ("NEG TV/AudioSet (test)", lambda: clips_split(W + "audioset_16k", "test")),
        ("NEG real female (test)", lambda: clips_split(W + "female_real_16k", "test")),
        ("NEG dinner party (test)", lambda: clips_split(W + "negative_datasets/dinner_party_wav", "test", 100) if os.path.isdir(W + "negative_datasets/dinner_party_wav") else iter(()))]
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
print("%-27s %-6s %5s  %s" % ("set", "mdl", "n", "  ".join(f"@{c}" for c in CUT)))
for label, _ in SETS:
    for name in MODELS:
        mx = res.get((name, label))
        if mx is None or not len(mx):
            continue
        if label.startswith("REAL"):
            print("%-27s %-6s %5d  %s" % (label, name, len(mx), "  ".join(f"{int((mx>=c).sum()):2d}/{len(mx):<2d}" for c in CUT)))
        else:
            print("%-27s %-6s %5d  %s" % (label, name, len(mx), "  ".join(f"{100*(mx>=c).mean():5.1f}%" for c in CUT)))
    print()
print("EVAL-V9-DONE")
