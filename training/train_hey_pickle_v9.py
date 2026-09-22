#!/usr/bin/env python
"""hey_pickle v9 (2026-09-13): REAL household positives (fine-tune) on top of the v6 recipe.

Data: real_clips2/ (Kellan_2 15 utterances, Ann_2 6) + real_clips3/ (Peggy_2 5, Leo_2 4), recorded
9/13 on a phone in the kitchen, each a mix of "hey pickle" and "ok pickle", split per utterance by
score_new.py. Validation/test = every 4th utterance per person (held out) unless MWW_REAL_TRAIN=all
(ship mode). The 8/25 memos are retired (Kellan: low quality, not needed).

Positive features: generated_augmented_features_real (train = real_clips2 x150 with the standard
augmentation: RIR, background music/TV noise, EQ, pitch; validation = real_clips_old x5; test = x1).
Training config = v6 negatives + synthetic positives (weight 2) + real positives (weight 4).

Modes (train stage, args: <name> <init|scratch> [lr] [steps]):
  python train_hey_pickle_v9.py data
  python train_hey_pickle_v9.py train v9_ft62a  v6_2  0.0003 3000   # fine-tune from v6.2 best weights
  python train_hey_pickle_v9.py train v9_ft56a  v5_6  0.0003 3000   # fine-tune from v5.6 (+ music_vocals negatives)
  python train_hey_pickle_v9.py train v9_s1     scratch 0.001 10000
Fine-tuning uses ~/mte_finetune.py (a copy of microwakeword.model_train_eval that loads
MWW_INIT_WEIGHTS after building the model).
"""
import os, sys, glob, subprocess, shutil
import numpy as np

WORK = os.path.expanduser("~/mww")
os.chdir(WORK)
STAGE = sys.argv[1] if len(sys.argv) > 1 else "data"
DEST = "/mnt/c/Users/kella/AppData/Local/Temp/claude/C--Users-kella-My-Drive--kellan-danielson-gmail-com--Projects-Home-Assistant/c080852f-3dff-415d-aed3-ca3a763fba14/scratchpad/"

def step(msg): print(f"\n===== {msg} =====", flush=True)

MODEL_ARGS = [
    "mixednet",
    "--pointwise_filters", "64,64,64,64", "--repeat_in_block", "1, 1, 1, 1",
    "--mixconv_kernel_sizes", "[5], [7,11], [9,15], [23]",
    "--residual_connection", "0,0,0,0", "--first_conv_filters", "32",
    "--first_conv_kernel_size", "5", "--stride", "3",
]
def train_args():
    return ["--training_config=training_parameters.yaml", "--train", "1", "--restore_checkpoint", "1",
            "--test_tf_nonstreaming", "0", "--test_tflite_nonstreaming", "0",
            "--test_tflite_nonstreaming_quantized", "0", "--test_tflite_streaming", "0",
            "--test_tflite_streaming_quantized", "1", "--use_weights", "best_weights"] + MODEL_ARGS

def write_yaml(train_dir, lr, steps):
    import yaml
    def neg(d, w):
        return {"features_dir": d, "sampling_weight": w, "penalty_weight": 1.0, "truth": False,
                "truncation_strategy": "random", "type": "mmap"}
    config = {
        "window_step_ms": 10, "train_dir": train_dir,
        "features": [
            {"features_dir": "generated_augmented_features_real", "sampling_weight": float(os.environ.get("MWW_REAL_W", "4.0")),
             "penalty_weight": 1.0, "truth": True, "truncation_strategy": "truncate_start", "type": "mmap"},
            {"features_dir": "generated_augmented_features_v5", "sampling_weight": 2.0,
             "penalty_weight": 1.0, "truth": True, "truncation_strategy": "truncate_start", "type": "mmap"},
            neg("negative_datasets/speech", 10.0), neg("negative_datasets/dinner_party", 10.0),
            neg("negative_datasets/no_speech", 5.0), neg("negative_datasets/tv_audio", 12.0),
            neg("negative_datasets/tts_negatives", 6.0), neg("negative_datasets/female_speech", 3.0),
            neg("negative_datasets/music_vocals", 10.0),
        ],
        "training_steps": [steps], "positive_class_weight": [1], "negative_class_weight": [int(os.environ.get("MWW_NEG_CW", "20"))],
        "learning_rates": [lr], "batch_size": 128,
        "time_mask_max_size": [0], "time_mask_count": [0], "freq_mask_max_size": [0], "freq_mask_count": [0],
        "eval_step_interval": 250, "clip_duration_ms": 1500, "target_minimization": 0.9,
        "minimization_metric": None, "maximization_metric": "average_viable_recall",
    }
    with open("training_parameters.yaml", "w") as f:
        yaml.dump(config, f)

if STAGE == "train":
    name = sys.argv[2]
    init = sys.argv[3] if len(sys.argv) > 3 else "scratch"
    lr = float(sys.argv[4]) if len(sys.argv) > 4 else (0.0003 if init != "scratch" else 0.001)
    steps = int(sys.argv[5]) if len(sys.argv) > 5 else (3000 if init != "scratch" else 10000)
    train_dir = f"trained_models/wakeword_{name}"
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    write_yaml(train_dir, lr, steps)
    env = dict(os.environ)
    if init != "scratch":
        env["MWW_INIT_WEIGHTS"] = os.path.abspath(f"trained_models/wakeword_{init}/best_weights.weights.h5")
        assert os.path.exists(env["MWW_INIT_WEIGHTS"]), env["MWW_INIT_WEIGHTS"]
        cmd = [sys.executable, os.path.expanduser("~/mte_finetune.py")] + train_args()
    else:
        cmd = [sys.executable, "-m", "microwakeword.model_train_eval"] + train_args()
    step(f"TRAIN {name} init={init} lr={lr} steps={steps}")
    subprocess.run(cmd, check=True, env=env)
    src = f"{train_dir}/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite"
    print("TFLITE:", os.path.abspath(src), os.path.getsize(src), "bytes")
    shutil.copy(src, DEST + f"hey_pickle_{name}.tflite")
    print(f"TRAIN-{name}-DONE")
    sys.exit(0)

# ---------------- data stage ----------------
import scipy.io.wavfile as wavfile
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

step("R1: real utterances -> real_clips_train (3 of 4 per person) + real_clips_val (every 4th)")
# Kellan 9/13: the 8/25 memos are retired (low quality). The 9/13 recordings are the dataset; each file was
# split into utterances (score_new.py). Validation/test = every 4th utterance per person, never trained on,
# unless MWW_REAL_TRAIN=all (ship mode: train on everything, validate on the same clips).
ALL_MODE = os.environ.get("MWW_REAL_TRAIN", "split") == "all"
for d in ("real_clips_train", "real_clips_val"):
    if os.path.isdir(d): shutil.rmtree(d)
    os.makedirs(d)
for d in ("real_clips2", "real_clips3"):
    for f in sorted(glob.glob(f"{d}/*.wav")):
        idx = int(os.path.basename(f).rsplit("_", 1)[1][:2])
        hold = (idx % 4 == 3) and not ALL_MODE
        os.symlink(os.path.abspath(f), os.path.join("real_clips_val" if hold else "real_clips_train", os.path.basename(f)))
        if ALL_MODE: os.symlink(os.path.abspath(f), os.path.join("real_clips_val", os.path.basename(f)))
TRAIN_SRC, VAL_SRC = "real_clips_train", "real_clips_val"
print("train:", len(glob.glob(TRAIN_SRC + "/*.wav")), "val:", len(glob.glob(VAL_SRC + "/*.wav")), "all-mode:", ALL_MODE)
assert len(glob.glob(TRAIN_SRC + "/*.wav")) >= 20

step("R2: real positive features -> generated_augmented_features_real")
from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration
from mmap_ninja.ragged import RaggedMmap
if os.path.exists("generated_augmented_features_real"):
    shutil.rmtree("generated_augmented_features_real")
augmenter = Augmentation(augmentation_duration_s=3.2,
    augmentation_probabilities={"SevenBandParametricEQ": 0.15, "TanhDistortion": 0.1, "PitchShift": 0.2,
        "BandStopFilter": 0.1, "AddColorNoise": 0.15, "AddBackgroundNoise": 0.75, "Gain": 1.0, "RIR": 0.5},
    impulse_paths=["mit_rirs"], background_paths=["fma_16k", "audioset_16k"],
    background_min_snr_db=-5, background_max_snr_db=12, min_jitter_s=0.195, max_jitter_s=0.205)
for split, src, rep, slide in [("training", TRAIN_SRC, 150, 10), ("validation", VAL_SRC, 5, 10), ("testing", VAL_SRC, 1, 1)]:
    clips = Clips(input_directory=src, file_pattern="*.wav", max_clip_duration_s=None, remove_silence=False, random_split_seed=None)
    out_dir = os.path.join("generated_augmented_features_real", split)
    os.makedirs(out_dir, exist_ok=True)
    spec = SpectrogramGeneration(clips=clips, augmenter=augmenter if split != "testing" else None, slide_frames=slide, step_ms=10)
    RaggedMmap.from_generator(out_dir=os.path.join(out_dir, "wakeword_mmap"),
        sample_generator=spec.spectrogram_generator(split=None, repeat=rep), batch_size=100, verbose=True)
    print(split, "done", flush=True)

step("R3: patched trainer -> ~/mte_finetune.py")
src = os.path.expanduser("~/microWakeWord/microwakeword/model_train_eval.py")
txt = open(src).read()
anchor = '        logging.info(model.summary())\n        train_model(config, model, data_processor, flags.restore_checkpoint)'
assert anchor in txt, "anchor not found in model_train_eval.py"
patched = txt.replace(anchor, '        _init = os.environ.get("MWW_INIT_WEIGHTS")\n        if _init:\n            model.load_weights(_init)\n            print("INIT WEIGHTS LOADED:", _init, flush=True)\n        logging.info(model.summary())\n        train_model(config, model, data_processor, flags.restore_checkpoint)')
open(os.path.expanduser("~/mte_finetune.py"), "w").write(patched)
print("DATA-STAGE-COMPLETE")
