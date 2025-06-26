CONFIGS=(
  "config/DeepseekV3.yaml"
  "config/DeepseekV3LPR.yaml"
  "config/Mixtral.yaml"
  "config/MixtralLPR.yaml"
  "config/Qwen3Moe.yaml"
  "config/Qwen3MoeLPR.yaml"
)
CHECKPOINTS=(
  "checkpoints/DeepseekV3-0.6B-0.1b/checkpoint-4000"
  "checkpoints/DeepseekV3LPR-0.6B-0.1b/checkpoint-4000"
  "checkpoints/Mixtral-0.6B-0.1b/checkpoint-4000"
  "checkpoints/MixtralLPR-0.6B-0.1b/checkpoint-4000"
  "checkpoints/Qwen3Moe0.6B-0.1b/checkpoint-4000"
  "checkpoints/Qwen3MoeLPR-0.6B-0.1b/checkpoint-4000"
)

if [ ${#CONFIGS[@]} -ne ${#CHECKPOINTS[@]} ]; then
  echo "ERROR: CONFIGS and CHECKPOINTS length mismatch"
  exit 1
fi

for i in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$i]}"
  ckpt="${CHECKPOINTS[$i]}"
  echo ">>> Starting evaluation with $config"
  python evaluate.py --config "$config" --checkpoint_path "$ckpt"
  echo ">>> Finished evaluation with $config"
done

echo ">>> All training tasks completed."
