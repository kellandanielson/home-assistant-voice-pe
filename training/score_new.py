import numpy as np, scipy.io.wavfile as wavfile, os, sys
from microwakeword.inference import Model
W = "/home/kella/mww/"
SC = "/mnt/c/Users/kella/AppData/Local/Temp/claude/C--Users-kella-My-Drive--kellan-danielson-gmail-com--Projects-Home-Assistant/c080852f-3dff-415d-aed3-ca3a763fba14/scratchpad/"
MODELS = {"v6.2": W + "trained_models/wakeword_v6_2/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite",
          "v5.6": W + "trained_models/wakeword_v5_6/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite",
          "v3": SC.replace("c080852f-3dff-415d-aed3-ca3a763fba14", "2635ef9b-b614-4916-95c9-60a058cdf7a8") + "hey_pickle_v3.tflite"}
def segments(a, frame=160, thr_ratio=0.10, min_gap=22, min_len=25, pad=15):
    n = len(a) // frame
    rms = np.array([np.sqrt(np.mean(a[i*frame:(i+1)*frame] ** 2)) for i in range(n)])
    thr = max(rms.max() * thr_ratio, 0.004)
    act = rms > thr
    segs = []; st = None; gap = 0
    for i, on in enumerate(act):
        if on:
            if st is None: st = i
            gap = 0
        elif st is not None:
            gap += 1
            if gap >= min_gap:
                e = i - gap
                if e - st >= min_len: segs.append((st, e))
                st = None; gap = 0
    if st is not None and n - st >= min_len: segs.append((st, n))
    return [(max(0, (s-pad)*frame), min(len(a), (e+pad)*frame)) for s, e in segs]
os.makedirs(W + "real_clips2", exist_ok=True)
for who in ("Ann_2", "Kellan_2"):
    sr, a = wavfile.read(SC + who + ".wav")
    a = (a.astype(np.float32) / 32767.0) if a.dtype == np.int16 else a.astype(np.float32)
    segs = segments(a)
    print(f"\n{who}: {len(a)/sr:.1f}s, peak {np.abs(a).max():.2f}, {len(segs)} utterances")
    ms = {k: Model(p) for k, p in MODELS.items() if os.path.exists(p)}
    for i, (s, e) in enumerate(segs):
        seg = a[s:e]
        wavfile.write(W + f"real_clips2/{who}_{i:02d}.wav", 16000, (seg * 32767).astype(np.int16))
        row = []
        for k, m in ms.items():
            p = np.asarray(m.predict_clip(seg), dtype=np.float32).ravel()
            row.append(f"{k}={p.max():.2f}")
            if hasattr(m, "reset"): m.reset()
        print(f"  {i:02d} {s/sr:5.1f}-{e/sr:5.1f}s ({(e-s)/sr:.1f}s) " + " ".join(row))
