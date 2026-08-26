#!/usr/bin/env python
"""v3 vs v4 vs v5 gate eval. Every set is held out from training (test splits
with the same random_split_seed, or entirely unseen voices). Reports the share
of clips whose max streaming probability crosses each tier cutoff.

Gates for shipping v5 (vs v3):
  unseen-voice positives (edge holdout incl. kid Maisie): >= v3
  TV (AudioSet test) crossings: <= v3          <- v4 never checked this
  real female speech crossings: < v3
  piper positives: ~= v3
Including v4 validates the eval itself: it must SHOW v4's unseen-voice failure.
"""
import glob, os
import numpy as np
from microwakeword.audio.clips import Clips
from microwakeword.inference import Model

MODELS = {
    "v3": "/mnt/c/Users/kella/AppData/Local/Temp/claude/C--Users-kella-My-Drive--kellan-danielson-gmail-com--Projects-Home-Assistant/2635ef9b-b614-4916-95c9-60a058cdf7a8/scratchpad/hey_pickle_v3.tflite",
    "v4": "/mnt/c/Users/kella/AppData/Local/Temp/claude/C--Users-kella-My-Drive--kellan-danielson-gmail-com--Projects-Home-Assistant/2635ef9b-b614-4916-95c9-60a058cdf7a8/scratchpad/hey_pickle_v4.tflite",
    "v5": "/home/kella/mww/trained_models/wakeword_v5/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite",
    "v5.1": "/home/kella/mww/trained_models/wakeword_v5_1/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite",
    "v5.2": "/home/kella/mww/trained_models/wakeword_v5_2/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite",
    "v5.3": "/home/kella/mww/trained_models/wakeword_v5_3/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite",
    "v5.4": "/home/kella/mww/trained_models/wakeword_v5_4/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite",
    "v5.5": "/home/kella/mww/trained_models/wakeword_v5_5/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite",
    "v5.6": "/home/kella/mww/trained_models/wakeword_v5_6/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite",
    "v5.7": "/home/kella/mww/trained_models/wakeword_v5_7/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite",
    "v5.8": "/home/kella/mww/trained_models/wakeword_v5_8/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite",
    "v5.9": "/home/kella/mww/trained_models/wakeword_v5_9/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite",
    "v5.10": "/home/kella/mww/trained_models/wakeword_v5_10/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite",
    "v5.11": "/home/kella/mww/trained_models/wakeword_v5_11/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite",
    "v5.12": "/home/kella/mww/trained_models/wakeword_v5_12/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite",
}
CUTOFFS = [0.45, 0.56, 0.85]
W = "/home/kella/mww/"
SETS = [  # (label, dir, split or None=all files, limit)
    ("POS piper (test)",        W + "generated_samples",     "test", None),
    ("POS edge train-voices (test)", W + "edge_pos_16k",     "test", None),
    ("POS edge UNSEEN voices",  W + "edge_pos_holdout_16k",  None,   None),
    ("POS edge UNSEEN kid (Maisie)", W + "edge_pos_holdout_16k", "maisie", None),
    ("NEG real female speech (test)", W + "female_real_16k", "test", None),
    ("NEG TV/AudioSet (test)",  W + "audioset_16k",          "test", None),
    ("NEG edge UNSEEN voices",  W + "edge_neg_holdout_16k",  None,   None),
    ("NEG tts train-voices (test)", W + "tts_neg_v5",        "test", None),
]

def clip_iter(d, split, limit):
    if split in (None, "maisie"):
        import scipy.io.wavfile as w
        files = sorted(glob.glob(os.path.join(d, "*.wav")))
        if split == "maisie": files = [f for f in files if "Maisie" in f]
        for f in files[:limit] if limit else files:
            sr, a = w.read(f)
            yield (a.astype(np.float32) / 32767.0) if a.dtype == np.int16 else a.astype(np.float32)
        return
    clips = Clips(input_directory=d, file_pattern="*.wav", max_clip_duration_s=None,
                  remove_silence=False, random_split_seed=10, split_count=0.1)
    n = 0
    for audio in clips.audio_generator(split=split, repeat=1):
        yield audio
        n += 1
        if limit and n >= limit: break

results = {}
for name, path in MODELS.items():
    if not os.path.exists(path):
        print(f"{name}: model missing, skipped"); continue
    m = Model(path)
    for label, d, split, limit in SETS:
        maxes = []
        try:
            for audio in clip_iter(d, split, limit):
                preds = m.predict_clip(audio)
                maxes.append(float(np.max(preds)) if len(preds) else 0.0)
                if hasattr(m, "reset"): m.reset()
        except Exception as e:
            print(f"{name} {label}: ERROR {str(e)[:120]}"); continue
        results[(name, label)] = np.array(maxes)

print("\n%-34s %-4s %6s  %s" % ("set", "mdl", "n", "  ".join(f"@{c}" for c in CUTOFFS)))
for label, *_ in SETS:
    for name in MODELS:
        mx = results.get((name, label))
        if mx is None or not len(mx): continue
        cells = "  ".join(f"{100*(mx>=c).mean():5.1f}%" for c in CUTOFFS)
        print("%-34s %-4s %6d  %s" % (label, name, len(mx), cells))
    print()
print("EVAL-DONE")
