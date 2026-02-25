# Fine-tuning, Evaluation, and Synthetic Data Generation for LLM Tool Calling with NVIDIA NeMo Microservices

## Introduction

Tool calling enables Large Language Models (LLMs) to interact with external systems, execute programs, and access real-time information unavailable in their training data. This capability allows LLMs to process natural language queries, map them to specific functions or APIs, and populate required parameters from user inputs. It's essential for building AI agents capable of tasks like checking inventory, retrieving weather data, managing workflows, and more.

### How LLM Tool Calling Works

- **Tools**: A function or tool refers to external functionality provided by the user to the model. As the model generates a response to the prompt, it may decide (or can be told) to use a tool to respond. In a real-world use case, you might provide tools to get weather for a location, access account details for a given user ID, or issue refunds for lost orders — scenarios where the LLM needs real-time information beyond its pretrained knowledge.

- **Tool calls**: A function call or tool call refers to a model's response when it decides it needs to call one of the available tools. For example, if a user sends "What's the weather in Paris?", the model responds with a tool call for the **get_weather** tool with **Paris** as the **location** argument.

- **Tool call outputs**: The response a tool generates using the input from a model's tool call. This can be structured JSON or plain text. For instance, the **get_weather** tool might return `{"temperature": "25", "unit": "C"}`, and the model then generates a natural language response like "The weather in Paris today is 25°C."

**Note:** Common failure patterns include malformed formats which end up in the model response instead of proper tool call structures.

<div style="text-align: center;">
<img src="./img/tool-use.png" alt="Example of a single-turn function call" width="60%" />
<p><strong>Figure 1:</strong> Example of a single-turn function call.</p>
</div>

### Customizing LLMs for Tool Calling

To effectively perform tool calling, an LLM must:

- Select the correct function(s) from a set of available options
- Extract and populate the appropriate parameters from a user's natural language query
- In multi-turn and multi-step use cases, plan and chain multiple actions together

As the number of tools and their complexity increases, customization becomes critical. Smaller models like **Nemotron Nano (8B)** can achieve accuracy comparable to much larger models through parameter-efficient techniques like [LoRA](https://arxiv.org/abs/2106.09685).

#### Tool calling with base model (before fine-tuning)

```bash
curl "$NEMO_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/nemotron-nano-llama-3.1-8b",
    "messages": [{"role": "user", "content": "What will the weather be in Berlin on November 7, 2025?"}],
    "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a given location and date.", "parameters": {"type": "object", "properties": {"location": {"type": "string"}, "date": {"type": "string"}}, "required": ["location", "date"]}}}],
    "tool_choice": "auto"
  }' | jq
```

**Note:** Without fine-tuning, the base model may return malformed responses or miss required arguments.

#### Tool calling after LoRA fine-tuning

After fine-tuning, the model correctly selects the function and extracts all arguments:

```json
{
  "tool_calls": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "arguments": "{\"location\": \"Berlin\", \"date\": \"2025-11-07\"}"
    }
  }]
}
```

### About NVIDIA NeMo Microservices

[NVIDIA NeMo Microservices](https://www.nvidia.com/en-us/ai-data-science/products/nemo/) are an API-first modular set of tools for customizing, evaluating, and securing LLMs. This tutorial uses the **v2 API** with workspace-scoped operations, filesets for data management, and the Python SDK's typed constructors.

## Objectives

This end-to-end tutorial demonstrates the **data flywheel** for tool calling: measure baseline accuracy → generate targeted synthetic data → fine-tune → verify improvement → deploy safely.

<div style="text-align: center;">
  <img src="./img/end-to-end-diagram.png" alt="End to End architecture" width="90%" />
  <p><strong>Figure 2:</strong> End-to-end architecture with NeMo Microservices, Data Designer, and NIM.</p>
</div>

The following notebooks are included:

1. [**Data Preparation and Baseline**](./1_data_preparation.ipynb) — Download xLAM dataset, convert to OpenAI format, upload to filesets, measure baseline accuracy
2. [**Synthetic Data with Data Designer**](./2_data_designer.ipynb) — Generate diverse, high-quality tool calling training data using NMP Data Designer
3. [**Fine-Tuning and Inference**](./3_finetuning_and_inference.ipynb) — LoRA fine-tune Nemotron Nano and test inference
4. [**Model Evaluation**](./4_model_evaluation.ipynb) — Compare baseline vs fine-tuned accuracy
5. [**Safety Guardrails**](./5_adding_safety_guardrails.ipynb) — Add content safety guardrails for production deployment

## Prerequisites

### Deploy NeMo Microservices

You will need NVIDIA GPUs allocated as follows:

- **Fine-tuning:** GPU(s) for fine-tuning Nemotron Nano (8B) using NeMo Customizer
- **Inference:** GPU(s) for deploying the Nemotron Nano NIM
- **(Optional)** Additional GPU for the content safety NIM (or use [build.nvidia.com](https://build.nvidia.com/))

Refer to the [platform prerequisites and installation guide](https://docs.nvidia.com/nemo/microservices/latest/get-started/platform-prereq.html) to deploy NeMo Microservices.

### Deploy Nemotron Nano NIM

Deploy the Nemotron Nano NIM for inference. Refer to the [NIM deployment instructions](https://docs.nvidia.com/nemo/microservices/latest/get-started/tutorials/deploy-nims.html) for details.

### Get Access to the xLAM Dataset

- Go to [xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) and request access (granted instantly)
- Obtain your [Hugging Face access token](https://huggingface.co/docs/hub/en/security-tokens)

## Get Started

1. Create a virtual environment:

   ```bash
   python3 -m venv nemo_env
   source nemo_env/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Update [config.py](./config.py) with your NMP deployment URLs:

   ```python
   NEMO_URL = "http://nemo.test"    # NeMo Microservices platform
   NIM_URL = "http://nim.test"      # NIM inference endpoint
   WORKSPACE = "default"            # Workspace name
   ```

4. Launch Jupyter Lab:

   ```bash
   jupyter lab --ip 0.0.0.0 --port=8888 --allow-root
   ```

5. Start with [1_data_preparation.ipynb](./1_data_preparation.ipynb).

## Other Notes

### About NVIDIA NIM

- This workflow is tailored to work with NVIDIA NIM for inference. It won't work with other inference providers (e.g., vLLM, SGLang, TGI).
- For improved inference speeds, use NIM with `fast_outlines` guided decoding. This is the default when deployed via the NeMo Microservices Helm Chart. If NIM is deployed separately, set `NIM_GUIDED_DECODING_BACKEND=fast_outlines`.

### Limitations with Tool Calling

If you use your own dataset or implement a different data preparation approach:
- Tool calls might take over 30 seconds if descriptions for `array` types lack `items` specifications, or if `object` types lack `properties` specifications. Include these details in tool descriptions.
- Tool calls may freeze the NIM if a tool description includes a function with more than 8 parameters. Ensure functions use 8 or fewer parameters.
