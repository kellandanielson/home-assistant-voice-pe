#!/usr/bin/env python
"""Real-voice gate: Ann + Peggy phone memos saying 'hey pickle'.
Splits each memo into utterances by energy (RMS over 10ms frames, silence
>= 400ms separates), runs each streaming model over every utterance, and
reports per-utterance max probability + how many would wake per tier."""
import glob, os, subprocess, sys
import numpy as np
import scipy.io.wavfile as wavfile
from microwakeword.inference import Model

UP = "/mnt/c/Users/kella/.claude/uploads/2635ef9b-b614-4916-95c9-60a058cdf7a8/"
OUT = "/home/kella/mww/real_clips/"
os.makedirs(OUT, exist_ok=True)
MODELS = {
    "v3":   "/mnt/c/Users/kella/AppData/Local/Temp/claude/C--Users-kella-My-Drive--kellan-danielson-gmail-com--Projects-Home-Assistant/2635ef9b-b614-4916-95c9-60a058cdf7a8/scratchpad/hey_pickle_v3.tflite",
    "v5":   "/home/kella/mww/trained_models/wakeword_v5/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite",
    "v5.1": "/home/kella/mww/trained_models/wakeword_v5_1/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite",
    "v5.4": "/home/kella/mww/trained_models/wakeword_v5_4/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite",
    "v5.6": "/home/kella/mww/trained_models/wakeword_v5_6/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite",
    "v5.7": "/home/kella/mww/trained_models/wakeword_v5_7/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite",
    "v5.8": "/home/kella/mww/trained_models/wakeword_v5_8/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite",
    "v5.9": "/home/kella/mww/trained_models/wakeword_v5_9/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite",
    "v5.10": "/home/kella/mww/trained_models/wakeword_v5_10/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite",
    "v5.11": "/home/kella/mww/trained_models/wakeword_v5_11/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite",
    "v5.12": "/home/kella/mww/trained_models/wakeword_v5_12/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite",
}
CUTOFFS = [0.45, 0.56, 0.65, 0.70, 0.75, 0.85]

def segments(a, sr=16000, frame=160, thr_ratio=0.10, min_gap_frames=22, min_len_frames=25, pad_frames=15):
    n = len(a) // frame
    rms = np.array([np.sqrt(np.mean(a[i*frame:(i+1)*frame] ** 2)) for i in range(n)])
    thr = max(rms.max() * thr_ratio, 0.004)
    active = rms > thr
    segs, start, gap = [], None, 0
    for i, on in enumerate(active):
        if on:
            if start is None: start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap_frames:
                end = i - gap
                if end - start >= min_len_frames: segs.append((start, end))
                start, gap = None, 0
    if start is not None and n - start >= min_len_frames: segs.append((start, n))
    out = []
    for s, e in segs:
        s0 = max(0, (s - pad_frames) * frame); e0 = min(len(a), (e + pad_frames) * frame)
        out.append(a[s0:e0])
    return out

clips = {}
for f in sorted(glob.glob(UP + "*.m4a")):
    who = os.path.basename(f).split("-", 1)[-1].replace(".m4a", "")
    wav = OUT + who + ".wav"
    if not os.path.exists(wav):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f, "-ar", "16000", "-ac", "1", wav], check=True)
    sr, a = wavfile.read(wav)
    a = a.astype(np.float32) / 32767.0 if a.dtype == np.int16 else a.astype(np.float32)
    segs = segments(a)
    clips[who] = (a, segs)
    print(f"{who}: {len(a)/sr:.1f}s, {len(segs)} utterances: " + ", ".join(f"{len(s)/sr:.1f}s" for s in segs))

for name, path in MODELS.items():
    if not os.path.exists(path):
        print(f"\n{name}: missing"); continue
    m = Model(path)
    print(f"\n=== {name} ===")
    for who, (a, segs) in clips.items():
        maxes = []
        for s in segs:
            p = np.asarray(m.predict_clip(s), dtype=np.float32).ravel()
            maxes.append(float(np.max(p)) if len(p) else 0.0)
            if hasattr(m, "reset"): m.reset()
        wakes = "  ".join(f"@{c}: {sum(x >= c for x in maxes)}/{len(maxes)}" for c in CUTOFFS)
        print(f"  {who:6s} {wakes}   maxes: " + " ".join(f"{x:.2f}" for x in maxes))
        # whole-clip streaming pass: count distinct crossings at 0.56 (what the puck would do on the raw memo)
        p = np.asarray(m.predict_clip(a), dtype=np.float32).ravel()
        if hasattr(m, "reset"): m.reset()
        cr = {c: int(np.sum((p[1:] >= c) & (p[:-1] < c))) if len(p) > 1 else 0 for c in CUTOFFS}
        print(f"         whole-memo distinct crossings: " + "  ".join(f"@{c}: {cr[c]}" for c in CUTOFFS) + f"   (peak {p.max():.2f})")
print("\nREAL-EVAL-DONE")
