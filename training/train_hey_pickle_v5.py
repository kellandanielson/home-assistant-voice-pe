#!/usr/bin/env python
"""hey_pickle v5 (2026-08-25): VOICE-DIVERSE, VOICE-BALANCED retrain.

v4 post-mortem: female-ONLY negatives vs all-voice positives let the model use
voice identity as a shortcut ("female voice -> negative"); a real woman saying
"hey pickle" at the puck stopped waking it, and the two new weight-8 negative
sets diluted tv_audio so TV false-wakes returned. Rolled back to v3 8/25.

v5 design (Kellan: generic kid/girl/boy/women voices, guests use it too):
  * Positives from TWO engines: piper (3000 clips, 904 adult LibriTTS voices,
    cached from v2/v3) + Microsoft edge-tts neural voices (47 English voices
    incl. child voices Ana/Maisie; rate/pitch prosody sweep; kid voice
    oversampled). A second engine stops the model keying on piper artifacts.
  * Negatives BALANCED BY VOICE: the SAME voices (piper all-speaker + the same
    edge voices) speaking 40 conversational/near-miss phrases -> the only
    remaining cue is the phrase. v4's female-only TTS set is DROPPED. The real
    LibriSpeech female set stays at a low weight (acoustic diversity).
  * tv_audio weight raised 8 -> 12 (v4 lesson: new sets re-balance old ones).
  * HELD-OUT edge voices (never trained) = the "unseen guest" FRR/FA gate.

Usage: python train_hey_pickle_v5.py data | train
Idempotent; workdir ~/mww; new outputs live in *_v5 / edge_* / piper_neg_* dirs.
"""
import os, sys, glob, json, random, subprocess, asyncio
from pathlib import Path

WORK = os.path.expanduser("~/mww")
PIPER = os.path.expanduser("~/piper-sample-generator")
PIPER_PY = os.path.expanduser("~/piper-venv/bin/python")
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)
STAGE = sys.argv[1] if len(sys.argv) > 1 else "data"

def step(msg):
    print(f"\n===== {msg} =====", flush=True)

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

if STAGE == "train":
    step("TRAIN v5")
    subprocess.run([sys.executable, "-m", "microwakeword.model_train_eval"] + TRAIN_ARGS, check=True)
    src = "trained_models/wakeword_v5/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite"
    print("TFLITE:", os.path.abspath(src), os.path.getsize(src), "bytes")
    import shutil
    dest = "/mnt/c/Users/kella/AppData/Local/Temp/claude/C--Users-kella-My-Drive--kellan-danielson-gmail-com--Projects-Home-Assistant/2635ef9b-b614-4916-95c9-60a058cdf7a8/scratchpad/hey_pickle_v5.tflite"
    shutil.copy(src, dest)
    print("COPIED TO:", dest)
    sys.exit(0)

# ---------------- data stage ----------------
PHRASES = [l.strip() for l in open("female_neg_phrases.txt") if l.strip()]  # 40 phrases from v4 (gender-neutral text)
assert len(PHRASES) == 40, len(PHRASES)

HOLDOUT_VOICES = ["en-GB-MaisieNeural", "en-AU-NatashaNeural", "en-US-MichelleNeural",
                  "en-IE-ConnorNeural", "en-CA-LiamNeural", "en-NZ-MollyNeural"]
KID_VOICES = ["en-US-AnaNeural"]  # Maisie is the held-out kid
POS_TEXTS = ["hey pickle", "hey pickle!", "Hey, pickle", "hey pickle?"]
RATES = ["-25%", "-10%", "+0%", "+15%", "+30%"]
PITCHES = ["-15Hz", "+0Hz", "+15Hz", "+35Hz"]

step("E1: edge-tts voice list")
import edge_tts
async def _voices():
    vs = await edge_tts.list_voices()
    return sorted(v["ShortName"] for v in vs if v["Locale"].startswith("en-"))
ALL_VOICES = asyncio.run(_voices())
TRAIN_VOICES = [v for v in ALL_VOICES if v not in HOLDOUT_VOICES]
print(len(ALL_VOICES), "en voices;", len(TRAIN_VOICES), "train;", len(HOLDOUT_VOICES), "holdout")
json.dump({"train": TRAIN_VOICES, "holdout": HOLDOUT_VOICES}, open("edge_voices_v5.json", "w"), indent=1)

async def synth_many(jobs, out_dir, concurrency=6):
    """jobs: list of (name, text, voice, rate, pitch). Skips existing wavs. mp3 -> 16k mono wav via ffmpeg."""
    os.makedirs(out_dir, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    done = fail = 0
    async def one(name, text, voice, rate, pitch):
        nonlocal done, fail
        wav = os.path.join(out_dir, name + ".wav")
        if os.path.exists(wav):
            done += 1; return
        mp3 = os.path.join(out_dir, name + ".mp3")
        async with sem:
            for attempt in range(3):
                try:
                    await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(mp3)
                    break
                except Exception as e:
                    if attempt == 2:
                        fail += 1; print("FAIL", name, str(e)[:80]); return
                    await asyncio.sleep(2 + attempt * 3)
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp3, "-ar", "16000", "-ac", "1", wav])
        if os.path.exists(mp3): os.remove(mp3)
        if r.returncode != 0: fail += 1
        else:
            done += 1
            if done % 200 == 0: print(f"  {done} clips", flush=True)
    await asyncio.gather(*(one(*j) for j in jobs))
    print(f"{out_dir}: {done} ok, {fail} failed")

rng = random.Random(10)
def pos_jobs(voices, per_voice, tag):
    jobs = []
    for v in voices:
        n = per_voice * (3 if v in KID_VOICES else 1)
        combos = [(t, r, p) for t in POS_TEXTS for r in RATES for p in PITCHES]
        rng.shuffle(combos)
        for i, (t, r, p) in enumerate(combos[:n]):
            jobs.append((f"{tag}_{v}_{i:03d}", t, v, r, p))
    return jobs
def neg_jobs(voices, tag):
    jobs = []
    for v in voices:
        for i, ph in enumerate(PHRASES):
            jobs.append((f"{tag}_{v}_{i:03d}", ph, v, rng.choice(RATES), rng.choice(PITCHES)))
    return jobs

step("E2: edge-tts positives (train voices ~50/voice, kid x3)")
if len(glob.glob("edge_pos_16k/*.wav")) < 1900:
    asyncio.run(synth_many(pos_jobs(TRAIN_VOICES, 50, "pos"), "edge_pos_16k"))
else: print("skip:", len(glob.glob("edge_pos_16k/*.wav")))

step("E3: edge-tts positives, HELD-OUT voices (eval only)")
if len(glob.glob("edge_pos_holdout_16k/*.wav")) < 150:
    asyncio.run(synth_many(pos_jobs(HOLDOUT_VOICES, 30, "hpos"), "edge_pos_holdout_16k"))
else: print("skip")

step("E4: edge-tts negatives, same train voices x 40 phrases")
if len(glob.glob("edge_neg_16k/*.wav")) < 1500:
    asyncio.run(synth_many(neg_jobs(TRAIN_VOICES, "neg"), "edge_neg_16k"))
else: print("skip")

step("E5: edge-tts negatives, held-out voices (eval only)")
if len(glob.glob("edge_neg_holdout_16k/*.wav")) < 200:
    asyncio.run(synth_many(neg_jobs(HOLDOUT_VOICES, "hneg"), "edge_neg_holdout_16k"))
else: print("skip")

step("P1: piper ALL-voice negatives (3000, replaces v4's female-only set)")
if len(glob.glob("piper_neg_16k/*.wav")) < 3000:
    subprocess.run([PIPER_PY, f"{PIPER}/generate_samples.py", os.path.join(WORK, "female_neg_phrases.txt"),
        "--max-samples", "3000", "--batch-size", "100",
        "--length-scales", "0.7", "0.85", "1.0", "1.15", "1.3",
        "--model", f"{PIPER}/models/en_US-libritts_r-medium.pt",
        "--output-dir", f"{WORK}/piper_neg_16k"], check=True, cwd=PIPER)
else: print("skip")

step("C1: combined dirs (symlinks): positives_v5, tts_neg_v5")
def link_all(dst, srcs):
    os.makedirs(dst, exist_ok=True)
    n = 0
    for s in srcs:
        for f in glob.glob(os.path.join(WORK, s, "*.wav")):
            l = os.path.join(dst, f"{s}__{os.path.basename(f)}")
            if not os.path.exists(l): os.symlink(f, l)
            n += 1
    print(dst, n, "clips")
link_all("positives_v5", ["generated_samples", "edge_pos_16k"])
link_all("tts_neg_v5", ["piper_neg_16k", "edge_neg_16k"])

step("F1: positive features (augmented) -> generated_augmented_features_v5")
from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration
from mmap_ninja.ragged import RaggedMmap

if not os.path.exists("generated_augmented_features_v5"):
    clips = Clips(input_directory="positives_v5", file_pattern="*.wav", max_clip_duration_s=None,
                  remove_silence=False, random_split_seed=10, split_count=0.1)
    augmenter = Augmentation(augmentation_duration_s=3.2,
        augmentation_probabilities={
            "SevenBandParametricEQ": 0.1, "TanhDistortion": 0.1, "PitchShift": 0.2,
            "BandStopFilter": 0.1, "AddColorNoise": 0.1, "AddBackgroundNoise": 0.75,
            "Gain": 1.0, "RIR": 0.5},
        impulse_paths=["mit_rirs"], background_paths=["fma_16k", "audioset_16k"],
        background_min_snr_db=-5, background_max_snr_db=10,
        min_jitter_s=0.195, max_jitter_s=0.205)
    for split, split_name, rep, slide in [("training", "train", 2, 10), ("validation", "validation", 1, 10), ("testing", "test", 1, 1)]:
        out_dir = os.path.join("generated_augmented_features_v5", split)
        os.makedirs(out_dir, exist_ok=True)
        spec = SpectrogramGeneration(clips=clips, augmenter=augmenter, slide_frames=slide, step_ms=10)
        RaggedMmap.from_generator(out_dir=os.path.join(out_dir, "wakeword_mmap"),
            sample_generator=spec.spectrogram_generator(split=split_name, repeat=rep),
            batch_size=100, verbose=True)
else: print("skip")

step("F2: tts_negatives features (voice-balanced)")
if not os.path.exists("negative_datasets/tts_negatives"):
    clips = Clips(input_directory="tts_neg_v5", file_pattern="*.wav", max_clip_duration_s=None,
                  remove_silence=False, random_split_seed=10, split_count=0.1)
    for split, split_name in [("training", "train"), ("validation", "validation"), ("testing", "test")]:
        out_dir = os.path.join("negative_datasets/tts_negatives", split)
        os.makedirs(out_dir, exist_ok=True)
        spec = SpectrogramGeneration(clips=clips, augmenter=None, slide_frames=1, step_ms=10)
        RaggedMmap.from_generator(out_dir=os.path.join(out_dir, "wakeword_mmap"),
            sample_generator=spec.spectrogram_generator(split=split_name, repeat=1),
            batch_size=50, verbose=True)
else: print("skip")

step("Y: training_parameters.yaml (v5)")
import yaml
config = {
    "window_step_ms": 10,
    "train_dir": "trained_models/wakeword_v5",
    "features": [
        {"features_dir": "generated_augmented_features_v5", "sampling_weight": 2.0,
         "penalty_weight": 1.0, "truth": True, "truncation_strategy": "truncate_start", "type": "mmap"},
        {"features_dir": "negative_datasets/speech", "sampling_weight": 10.0,
         "penalty_weight": 1.0, "truth": False, "truncation_strategy": "random", "type": "mmap"},
        {"features_dir": "negative_datasets/dinner_party", "sampling_weight": 10.0,
         "penalty_weight": 1.0, "truth": False, "truncation_strategy": "random", "type": "mmap"},
        {"features_dir": "negative_datasets/no_speech", "sampling_weight": 5.0,
         "penalty_weight": 1.0, "truth": False, "truncation_strategy": "random", "type": "mmap"},
        # v3 TV fix, weight raised 8 -> 12 (v4 diluted it)
        {"features_dir": "negative_datasets/tv_audio", "sampling_weight": 12.0,
         "penalty_weight": 1.0, "truth": False, "truncation_strategy": "random", "type": "mmap"},
        # v5: voice-balanced TTS negatives (same voices as positives, both engines)
        {"features_dir": "negative_datasets/tts_negatives", "sampling_weight": 6.0,
         "penalty_weight": 1.0, "truth": False, "truncation_strategy": "random", "type": "mmap"},
        # real female speech kept at LOW weight for acoustic diversity (was 8 in v4)
        {"features_dir": "negative_datasets/female_speech", "sampling_weight": 3.0,
         "penalty_weight": 1.0, "truth": False, "truncation_strategy": "random", "type": "mmap"},
    ],
    "training_steps": [10000], "positive_class_weight": [1], "negative_class_weight": [20],
    "learning_rates": [0.001], "batch_size": 128,
    "time_mask_max_size": [0], "time_mask_count": [0], "freq_mask_max_size": [0], "freq_mask_count": [0],
    "eval_step_interval": 500, "clip_duration_ms": 1500, "target_minimization": 0.9,
    "minimization_metric": None, "maximization_metric": "average_viable_recall",
}
with open("training_parameters.yaml", "w") as f:
    yaml.dump(config, f)
print("DATA-STAGE-COMPLETE")
