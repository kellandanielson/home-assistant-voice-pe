#!/usr/bin/env python
"""hey_pickle v8 (2026-09-12): v6.2 recipe + "OK PICKLE" POSITIVES + "okay" near-miss negatives.

Why: Kellan 9/12 - "hey pickle AND ok pickle is usually what people say", and waking the
puck is "extremely difficult for every household member". Every model so far (v3..v7)
was trained ONLY on "hey pickle" (POS_TEXTS in v5). "ok pickle" wakes it by accident.

v8 = v6 data (all cached) +
  * positives: piper "okay pickle"/"ok pickle" (3000, same libritts model + length scales
    as the v3 "hey pickle" set) + edge-tts ok-variants on the SAME train voices
    (~50/voice, kid x3) -> positives_v8 = old positives + new -> generated_augmented_features_v8
  * negatives: 12 "okay ..." near-miss phrases on the same edge voices + piper (1000)
    -> tts_neg_v8 = tts_neg_v5 + new -> negative_datasets/tts_negatives_v8
  * held-out edge voices get "ok pickle" positives too (edge_okpos_holdout_16k) = unseen-voice gate.
Weights = v6. Household memos (the real v8 plan) still pending - fold in when recorded.

Usage: python train_hey_pickle_v8.py data | train <name>
Idempotent; workdir ~/mww.
"""
import os, sys, glob, json, random, subprocess, asyncio

WORK = os.path.expanduser("~/mww")
PIPER = os.path.expanduser("~/piper-sample-generator")
PIPER_PY = os.path.expanduser("~/piper-venv/bin/python")
os.chdir(WORK)
STAGE = sys.argv[1] if len(sys.argv) > 1 else "data"
DEST = "/mnt/c/Users/kella/AppData/Local/Temp/claude/C--Users-kella-My-Drive--kellan-danielson-gmail-com--Projects-Home-Assistant/c080852f-3dff-415d-aed3-ca3a763fba14/scratchpad/"

def step(msg): print(f"\n===== {msg} =====", flush=True)

TRAIN_ARGS = [
    "--training_config=training_parameters.yaml",
    "--train", "1", "--restore_checkpoint", "1",
    "--test_tf_nonstreaming", "0", "--test_tflite_nonstreaming", "0",
    "--test_tflite_nonstreaming_quantized", "0", "--test_tflite_streaming", "0",
    "--test_tflite_streaming_quantized", "1", "--use_weights", "best_weights",
    "mixednet",
    "--pointwise_filters", "64,64,64,64", "--repeat_in_block", "1, 1, 1, 1",
    "--mixconv_kernel_sizes", "[5], [7,11], [9,15], [23]",
    "--residual_connection", "0,0,0,0", "--first_conv_filters", "32",
    "--first_conv_kernel_size", "5", "--stride", "3",
]

def write_yaml(train_dir):
    import yaml
    def neg(d, w):
        return {"features_dir": d, "sampling_weight": w, "penalty_weight": 1.0, "truth": False,
                "truncation_strategy": "random", "type": "mmap"}
    config = {
        "window_step_ms": 10, "train_dir": train_dir,
        "features": [
            {"features_dir": "generated_augmented_features_v8", "sampling_weight": 2.0,
             "penalty_weight": 1.0, "truth": True, "truncation_strategy": "truncate_start", "type": "mmap"},
            neg("negative_datasets/speech", 10.0), neg("negative_datasets/dinner_party", 10.0),
            neg("negative_datasets/no_speech", 5.0), neg("negative_datasets/tv_audio", 12.0),
            neg("negative_datasets/tts_negatives_v8", 6.0), neg("negative_datasets/female_speech", 3.0),
            neg("negative_datasets/music_vocals", 10.0),
        ],
        "training_steps": [10000], "positive_class_weight": [1], "negative_class_weight": [20],
        "learning_rates": [0.001], "batch_size": 128,
        "time_mask_max_size": [0], "time_mask_count": [0], "freq_mask_max_size": [0], "freq_mask_count": [0],
        "eval_step_interval": 500, "clip_duration_ms": 1500, "target_minimization": 0.9,
        "minimization_metric": None, "maximization_metric": "average_viable_recall",
    }
    with open("training_parameters.yaml", "w") as f:
        yaml.dump(config, f)

if STAGE == "train":
    name = sys.argv[2] if len(sys.argv) > 2 else "v8_1"
    write_yaml(f"trained_models/wakeword_{name}")
    step(f"TRAIN {name}")
    subprocess.run([sys.executable, "-m", "microwakeword.model_train_eval"] + TRAIN_ARGS, check=True)
    src = f"trained_models/wakeword_{name}/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite"
    print("TFLITE:", os.path.abspath(src), os.path.getsize(src), "bytes")
    import shutil
    shutil.copy(src, DEST + f"hey_pickle_{name}.tflite")
    print(f"TRAIN-{name}-DONE")
    sys.exit(0)

# ---------------- data stage ----------------
V = json.load(open("edge_voices_v5.json"))
TRAIN_VOICES, HOLDOUT_VOICES = V["train"], V["holdout"]
KID_VOICES = ["en-US-AnaNeural"]
OK_TEXTS = ["ok pickle", "okay pickle", "OK, pickle", "okay pickle!", "ok pickle?"]
OK_NEG = ["okay", "okay okay okay", "ok let's go", "okay pick it up", "ok people listen up",
          "okay so what's next", "ok bye", "okay pick one", "ok tickle me", "okay nickel",
          "ok pickles are gross", "okay picking up the kids now"]
RATES = ["-25%", "-10%", "+0%", "+15%", "+30%"]
PITCHES = ["-15Hz", "+0Hz", "+15Hz", "+35Hz"]
open("ok_pos_phrases.txt", "w").write("okay pickle\nok pickle\n")
open("ok_neg_phrases.txt", "w").write("\n".join(OK_NEG) + "\n")

import edge_tts
async def synth_many(jobs, out_dir, concurrency=6):
    os.makedirs(out_dir, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    done = fail = 0
    async def one(name, text, voice, rate, pitch):
        nonlocal done, fail
        wav = os.path.join(out_dir, name + ".wav")
        if os.path.exists(wav):
            done += 1
            return
        mp3 = os.path.join(out_dir, name + ".mp3")
        async with sem:
            for attempt in range(3):
                try:
                    await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(mp3)
                    break
                except Exception as e:
                    if attempt == 2:
                        fail += 1
                        print("FAIL", name, str(e)[:80])
                        return
                    await asyncio.sleep(2 + attempt * 3)
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp3, "-ar", "16000", "-ac", "1", wav])
        if os.path.exists(mp3):
            os.remove(mp3)
        if r.returncode != 0:
            fail += 1
        else:
            done += 1
            if done % 200 == 0:
                print(f"  {done} clips", flush=True)
    await asyncio.gather(*(one(*j) for j in jobs))
    print(f"{out_dir}: {done} ok, {fail} failed", flush=True)

rng = random.Random(12)
def pos_jobs(voices, per_voice, tag):
    jobs = []
    for v in voices:
        n = per_voice * (3 if v in KID_VOICES else 1)
        combos = [(t, r, p) for t in OK_TEXTS for r in RATES for p in PITCHES]
        rng.shuffle(combos)
        for i, (t, r, p) in enumerate(combos[:n]):
            jobs.append((f"{tag}_{v}_{i:03d}", t, v, r, p))
    return jobs
def neg_jobs(voices, tag):
    return [(f"{tag}_{v}_{i:03d}", ph, v, rng.choice(RATES), rng.choice(PITCHES))
            for v in voices for i, ph in enumerate(OK_NEG)]

step("E1: edge-tts OK positives, train voices")
if len(glob.glob("edge_okpos_16k/*.wav")) < 1900:
    asyncio.run(synth_many(pos_jobs(TRAIN_VOICES, 50, "okpos"), "edge_okpos_16k"))
else:
    print("skip")
step("E2: edge-tts OK positives, HELD-OUT voices (eval only)")
if len(glob.glob("edge_okpos_holdout_16k/*.wav")) < 150:
    asyncio.run(synth_many(pos_jobs(HOLDOUT_VOICES, 30, "hokpos"), "edge_okpos_holdout_16k"))
else:
    print("skip")
step("E3: edge-tts OK near-miss negatives, train voices")
if len(glob.glob("edge_okneg_16k/*.wav")) < 400:
    asyncio.run(synth_many(neg_jobs(TRAIN_VOICES, "okneg"), "edge_okneg_16k"))
else:
    print("skip")
step("E4: edge-tts OK near-miss negatives, held-out voices (eval only)")
if len(glob.glob("edge_okneg_holdout_16k/*.wav")) < 60:
    asyncio.run(synth_many(neg_jobs(HOLDOUT_VOICES, "hokneg"), "edge_okneg_holdout_16k"))
else:
    print("skip")

step("P1: piper OK positives (3000)")
if len(glob.glob("piper_okpos_16k/*.wav")) < 2900:
    subprocess.run([PIPER_PY, f"{PIPER}/generate_samples.py", os.path.join(WORK, "ok_pos_phrases.txt"),
        "--max-samples", "3000", "--batch-size", "100", "--length-scales", "0.7", "0.85", "1.0", "1.15", "1.3",
        "--model", f"{PIPER}/models/en_US-libritts_r-medium.pt", "--output-dir", f"{WORK}/piper_okpos_16k"], check=True, cwd=PIPER)
else:
    print("skip")
step("P2: piper OK near-miss negatives (1000)")
if len(glob.glob("piper_okneg_16k/*.wav")) < 950:
    subprocess.run([PIPER_PY, f"{PIPER}/generate_samples.py", os.path.join(WORK, "ok_neg_phrases.txt"),
        "--max-samples", "1000", "--batch-size", "100", "--length-scales", "0.7", "0.85", "1.0", "1.15", "1.3",
        "--model", f"{PIPER}/models/en_US-libritts_r-medium.pt", "--output-dir", f"{WORK}/piper_okneg_16k"], check=True, cwd=PIPER)
else:
    print("skip")

step("C1: combined dirs: positives_v8, tts_neg_v8")
def link_all(dst, srcs):
    os.makedirs(dst, exist_ok=True)
    n = 0
    for s in srcs:
        for f in glob.glob(os.path.join(WORK, s, "*.wav")):
            l = os.path.join(dst, f"{s}__{os.path.basename(f)}")
            if not os.path.exists(l):
                os.symlink(f, l)
            n += 1
    print(dst, n, "clips")
link_all("positives_v8", ["generated_samples", "edge_pos_16k", "piper_okpos_16k", "edge_okpos_16k"])
link_all("tts_neg_v8", ["piper_neg_16k", "edge_neg_16k", "piper_okneg_16k", "edge_okneg_16k"])

from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration
from mmap_ninja.ragged import RaggedMmap
step("F1: positive features (augmented) -> generated_augmented_features_v8")
if not os.path.exists("generated_augmented_features_v8"):
    clips = Clips(input_directory="positives_v8", file_pattern="*.wav", max_clip_duration_s=None,
                  remove_silence=False, random_split_seed=10, split_count=0.1)
    augmenter = Augmentation(augmentation_duration_s=3.2,
        augmentation_probabilities={"SevenBandParametricEQ": 0.1, "TanhDistortion": 0.1, "PitchShift": 0.2,
            "BandStopFilter": 0.1, "AddColorNoise": 0.1, "AddBackgroundNoise": 0.75, "Gain": 1.0, "RIR": 0.5},
        impulse_paths=["mit_rirs"], background_paths=["fma_16k", "audioset_16k"],
        background_min_snr_db=-5, background_max_snr_db=10, min_jitter_s=0.195, max_jitter_s=0.205)
    for split, split_name, rep, slide in [("training", "train", 2, 10), ("validation", "validation", 1, 10), ("testing", "test", 1, 1)]:
        out_dir = os.path.join("generated_augmented_features_v8", split)
        os.makedirs(out_dir, exist_ok=True)
        spec = SpectrogramGeneration(clips=clips, augmenter=augmenter, slide_frames=slide, step_ms=10)
        RaggedMmap.from_generator(out_dir=os.path.join(out_dir, "wakeword_mmap"),
            sample_generator=spec.spectrogram_generator(split=split_name, repeat=rep), batch_size=100, verbose=True)
else:
    print("skip")
step("F2: tts_negatives_v8 features")
if not os.path.exists("negative_datasets/tts_negatives_v8"):
    clips = Clips(input_directory="tts_neg_v8", file_pattern="*.wav", max_clip_duration_s=None,
                  remove_silence=False, random_split_seed=10, split_count=0.1)
    for split, split_name in [("training", "train"), ("validation", "validation"), ("testing", "test")]:
        out_dir = os.path.join("negative_datasets/tts_negatives_v8", split)
        os.makedirs(out_dir, exist_ok=True)
        spec = SpectrogramGeneration(clips=clips, augmenter=None, slide_frames=1, step_ms=10)
        RaggedMmap.from_generator(out_dir=os.path.join(out_dir, "wakeword_mmap"),
            sample_generator=spec.spectrogram_generator(split=split_name, repeat=1), batch_size=50, verbose=True)
else:
    print("skip")
print("DATA-STAGE-COMPLETE")
