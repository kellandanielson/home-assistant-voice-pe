# hey_pickle training (v5 recipe, 2026-08-25)

- `train_hey_pickle_v5.py` — data + train stages (WSL2, RTX 4070). Voice-diverse
  positives (piper 904 voices + edge-tts 41 neural voices incl. child), negatives
  BALANCED BY VOICE (same voices, 40 phrases), tv_audio weight 12, held-out edge
  voices for unseen-voice eval. Runner scripts in WSL: ~/run_v5_{data,train}.sh.
- `run_v5_seeds.sh` — trains N identical-config seeds. REQUIRED: microWakeWord
  training at this size is a lottery (12 seeds ranged from deaf to trigger-happy).
  Never ship a single roll.
- `eval_v5.py` — synthetic gates (held-out piper/edge positives, unseen voices,
  unseen kid, real female speech, TV/AudioSet, unseen-voice negatives).
- `eval_real.py` — THE gate: real phone memos of the household saying "hey pickle"
  (Kellan/Ann/Peggy, WSL ~/mww/real_clips/*.wav). v4 and v5.4 both passed every
  synthetic gate and failed real voices. Phone-mic conditions depress absolute
  rates (v3 scores 0/8 on Peggy yet wakes for her) — compare models RELATIVELY.
- Shipped v5.6 at the Slightly tier (0.85): real wakes >= v3@Moderately, female-
  speech false-accepts ~halved, TV down a third. Moderately (0.56) = the more
  sensitive knob. `wakewords/hey_pickle.v5_12-candidate.tflite` = the "Ann wakes
  it from anywhere" seed (6/11 @0.85) but TV 13% — kept for future evaluation.
