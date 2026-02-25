# NeMo Microservices Platform URL (all services accessible via single endpoint in v2)
NEMO_URL = "http://nemo.test"

# NIM inference endpoint
NIM_URL = "http://nim.test"

# Workspace for scoping all resources
WORKSPACE = "default"

# Base model for fine-tuning
BASE_MODEL = "nvidia/nemotron-nano-llama-3.1-8b"
BASE_MODEL_URI = "hf://nvidia/nemotron-nano-llama-3.1-8b"

# Customization target and job naming
TARGET_NAME = "nemotron-nano-tool-calling"
JOB_NAME = "tool-calling-sft"

# Fileset names for data storage
TRAINING_FILESET = "xlam-training"
EVAL_FILESET = "xlam-eval"
DD_TRAINING_FILESET = "dd-tool-calling-training"

# Content safety model for guardrails
CONTENT_SAFETY_MODEL = "nvidia/llama-3.1-nemoguard-8b-content-safety"

# (Optional) To observe training with WandB
WANDB_API_KEY = ""
