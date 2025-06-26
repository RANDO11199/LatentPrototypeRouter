CONFIGS=(
  "config/DeepseekV3.yaml"
  "config/DeepseekV3LPR.yaml"
  "config/Mixtral.yaml"
  "config/MixtralLPR.yaml"
  "config/Qwen3Moe.yaml"
  "config/Qwen3MoeLPR.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo ">>> Starting training with $config"
  python train.py --config "$config"
  echo ">>> Finished training with $config"
done

echo ">>> All training tasks completed."
