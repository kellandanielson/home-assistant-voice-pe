#!/bin/bash
# v5.2 = exact v5 weights re-run (seed variance probe); v5.3 = positives 3, tts_negatives 6
source ~/mww-venv/bin/activate
export LD_LIBRARY_PATH=$(find ~/mww-venv/lib/python3.12/site-packages/nvidia -maxdepth 2 -name lib -type d | paste -sd:)
cd ~/mww
SP=/mnt/c/Users/kella/AppData/Local/Temp/claude/C--Users-kella-My-Drive--kellan-danielson-gmail-com--Projects-Home-Assistant/2635ef9b-b614-4916-95c9-60a058cdf7a8/scratchpad
train_one () {  # name pos_w tts_w
python - "$1" "$2" "$3" <<'PY'
import yaml, sys
name, pw, tw = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
c = yaml.safe_load(open("training_parameters.yaml"))
c["train_dir"] = f"trained_models/wakeword_{name}"
for f in c["features"]:
    if f["features_dir"] == "generated_augmented_features_v5": f["sampling_weight"] = pw
    if f["features_dir"] == "negative_datasets/tts_negatives": f["sampling_weight"] = tw
yaml.dump(c, open("training_parameters.yaml", "w"))
print("yaml", name, [(f["features_dir"], f["sampling_weight"]) for f in c["features"]], flush=True)
PY
python -m microwakeword.model_train_eval --training_config=training_parameters.yaml --train 1 --restore_checkpoint 1 \
  --test_tf_nonstreaming 0 --test_tflite_nonstreaming 0 --test_tflite_nonstreaming_quantized 0 --test_tflite_streaming 0 \
  --test_tflite_streaming_quantized 1 --use_weights best_weights mixednet --pointwise_filters 64,64,64,64 \
  --repeat_in_block "1, 1, 1, 1" --mixconv_kernel_sizes "[5], [7,11], [9,15], [23]" --residual_connection 0,0,0,0 \
  --first_conv_filters 32 --first_conv_kernel_size 5 --stride 3
cp trained_models/wakeword_$1/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite $SP/hey_pickle_$1.tflite
echo "TRAIN-$1-DONE"
}
train_one v5_2 2 6
train_one v5_3 3 6
echo SEEDS-ALL-DONE
