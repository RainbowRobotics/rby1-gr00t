# RBY1 GR00T

This repository adds RBY1 embodiment support on top of NVIDIA Isaac GR00T (N1.7-based).

For details on Isaac GR00T itself, refer to the upstream [Isaac GR00T README](https://github.com/NVIDIA/Isaac-GR00T/blob/n1.7-release/README.md).

## Overview

This repository adds RBY1 embodiment support to the Isaac GR00T training and inference pipeline — the dataset schema, modality configuration, and fine-tuning entry point needed to register and train RBY1M as a new embodiment in Isaac GR00T. On the real robot, the served policy connects over ZMQ to [`rby1-lerobot`](https://github.com/RainbowRobotics/rby1-lerobot).

<img src="rby1/rby1-gr00t_overview.png">

## Key Files

| Item | File |
| ---- | ---- |
| Dataset metadata file | `rby1/modality.json` |
| RBY1M data & modality configuration for Isaac GR00T | `rby1/rby1m_config.py` |
| Fine-tuning launch script | `rby1/rby1_finetune.sh` |
| Mixed-aspect-ratio camera patch | `gr00t/model/gr00t_n1d7/image_augmentations.py` |

The RBY1 integration includes:

* `modality.json` describing the RBY1 dataset schema (state/action/video key mapping), to be placed under `meta/modality.json`
* `rby1m_config.py` defining the RBY1M data/modality configuration and `NEW_EMBODIMENT` tag registration
* `rby1_finetune.sh` for launching fine-tuning on RBY1M data
* `image_augmentations.py` and `examples/rby1/finetune.sh` have been patched to support cameras with different aspect ratios in the same configuration (e.g. RBY1's landscape front camera at 480×640 vs. portrait wrist cameras at 640×480). This mismatch arises because `rby1-lerobot` allows each camera's mounting orientation to be configured independently. The original NVIDIA Isaac-GR00T repository assumes all camera views share the same aspect ratio and fails on such mixed setups; this patch removes that restriction. All modified locations are marked with `# [RBY1 PATCH]` in the source code.

These additions and command examples let you use the standard Isaac GR00T data preparation, fine-tuning, and inference workflow while connecting RBY1 datasets and RBY1 robot clients.

## Installation

### Clone the Repository

GR00T relies on submodules for certain dependencies. Include them when cloning:

**Note:** `git-lfs` is **required** to download parquet data files in `/demo_data`. Install it before cloning: 

```sh
sudo apt install git-lfs && git lfs install

git clone --recurse-submodules https://github.com/RainbowRobotics/rby1-gr00t
cd rby1-gr00t
```

If you've already cloned without submodules, initialize them separately: `git submodule update --init --recursive`

### Set Up the Environment

GR00T uses [uv](https://github.com/astral-sh/uv) for fast, reproducible dependency management. Install uv first:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### dGPU (x86_64) — Default

Install FFmpeg (required by `torchcodec`, the only supported video backend):
```sh
sudo apt-get update && sudo apt-get install -y ffmpeg
```

Create the environment and install GR00T:
```sh
uv sync --python 3.10
```
GPU dependencies (flash-attn, TensorRT, etc.) are included in the default install.

Verify the installation:
```sh
uv run python -c "import gr00t; print('GR00T installed successfully')"
```

> **Hugging Face access (required):** GR00T's VLM backbone is [`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B), a **gated** model that every GR00T checkpoint (including the base `nvidia/GR00T-N1.7-3B`) loads on first use. Before running inference or finetuning, request access on the model page and authenticate:
> ```sh
> uv run huggingface-cli login   # or: export HF_TOKEN=<your_token>
> ```
> Without access, model loading fails with a `GatedRepoError` / `401 Client Error`.

> **`flash-attn` message on every `uv run`:** You may see `Installing flash-attn...` each time you run `uv run`. This is a known `uv` behavior with URL-pinned wheel sources — `uv` re-validates the cached wheel against the source URL on each invocation. It is **not** rebuilding from source; the wheel is already cached locally and the operation takes 2-3 seconds. This affects platforms that use URL-pinned flash-attn wheels (x86_64 and aarch64). 
> To suppress it, remove the `flash-attn` entries under `[tool.uv.sources]` in your local `pyproject.toml` after the initial install. But that will break `uv lock` and cause flash-attn to build from source on next lock regeneration.

## Training

### 1. Prepare the dataset

To collect the dataset via teleoperation, please refer to the official documentation in [rby1-lerobot](https://github.com/RainbowRobotics/rby1-lerobot).

**1) Conversion** — Convert your dataset from **LeRobot v3** to **LeRobot v2**.

```bash
uv run --project scripts/lerobot_conversion \
  python scripts/lerobot_conversion/convert_v3_to_v2.py \
  --repo-id <HF_NAMESPACE>/<DATASET_NAME> \
  --root dataset
```
**2) Dataset meta/modality file** — Then move the `modality.json` file into the dataset's `meta/` directory.
```bash
cp rby1/modality.json dataset/<HF_NAMESPACE>/<DATASET_NAME>/meta/modality.json
```

> [!NOTE]
> - `rby1/modality.json` is an example for a stationary-base, left/right-arm robot (1-DoF gripper each) with three cameras (front/left/right) — if your gripper DoF, camera setup differs, or your setup includes a mobile (moving) base, write a new `modality.json` for your dataset, and make sure its `state`/`action` key/index ranges exactly match the index order in your dataset's `meta/info.json` (a mismatch won't raise an error, it will silently map values to the wrong joints).

**3) Dataset Statistics Generation** — Generate the statistics file expected by GR00T:

```bash
uv run python gr00t/data/stats.py \
  --dataset-path dataset/<HF_NAMESPACE>/<DATASET_NAME> \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path rby1/rby1m_config.py
```

> [!NOTE]
> - `--modality-config-path` : Its `state`/`action` key ranges must match the dataset's `meta/modality.json` — `rby1/rby1m_config.py` is prepared to match the `rby1/modality.json` above. Also double-check the EEF/non-EEF action types, and set `_ACTION_HORIZON` to the action horizon you want to use (this value determines both statistics generation and fine-tuning).

### 2. Start training

```bash
USE_WANDB=0 NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0 TUNE_VISUAL=0 uv run bash rby1/rby1_finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path dataset/<HF_NAMESPACE>/<DATASET_NAME> \
  --modality-config-path rby1/rby1m_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir outputs/<OUTPUT_DIR_NAME>
```

> [!NOTE] 
> - `TUNE_VISUAL=1` option uses more GPU memory
> - Use the same `--modality-config-path` as the one used for statistics generation

During training:

- Training logs are printed to the console.
- Checkpoints are automatically saved in the `checkpoint-<N>/` directory.
- If **Weights & Biases (W&B)** is enabled, training progress can also be monitored from the W&B dashboard.

## Offline Evaluation

Before deploying to the real robot, use offline evaluation to sanity-check the finetuned checkpoint against recorded trajectories. The evaluation produces visualizations comparing predicted actions against ground truth trajectories.  

Evaluate the finetuned model with the following command:

```bash
uv run python gr00t/eval/open_loop_eval.py \
  --dataset-path dataset/<HF_NAMESPACE>/<DATASET_NAME> \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path outputs/<OUTPUT_DIR_NAME>/checkpoint-<N> \
  --traj-ids 0 \
  --action-horizon 40 \
  --steps 400 \
  --save-plot-path outputs/<OUTPUT_DIR_NAME>/<PLOT_NAME>
```

## Deploying to the Real Robot (via rby1-lerobot)

You can use the ZMQ-based policy server to communicate with an RBY1 robot client.

For example, when using the RBY1 robot client implemented in `rby1-lerobot`, run the `rby1-gr00t` policy server as follows:

**Terminal 1 — Start the policy server:**

```bash
uv run python gr00t/eval/run_gr00t_server.py \
  --host 0.0.0.0 --port 5555 \
  --model-path outputs/<OUTPUT_DIR_NAME>/checkpoint-<N>
```
> [!NOTE] 
> - This `rby1-gr00t` patched repository must be used for **both fine-tuning and inference**.
> - `0.0.0.0` is not a real address; it's a *bind setting* meaning "the server accepts
> connections on all of its network interfaces."  
> - If you get ZMQError: Address already in use, the default port 5555 is occupied. Use --port <other_port>.
> - <OUTPUT_DIR_NAME> must point to the actual trained checkpoint path.

**Terminal 2 — Run RBY1 LeRobot client:**

After that, configure the RBY1 robot client to connect to the same host and port. The client can then receive action chunks from the GR00T model and use them for robot control.


For the RBY1 LeRobot client, see [rby1-lerobot](https://github.com/RainbowRobotics/rby1-lerobot).


Install the client on the UPC:

```bash
git clone https://github.com/RainbowRobotics/rby1-lerobot.git
cd rby1-lerobot
pip install -e lerobot-robot-rby1
pip install -e lerobot-async-rby1
```

Start the robot client:

```bash
lerobot-robot-client \
  --backend=groot_zmq --robot.type=rby1 \
  --server_address=<POLICY_SERVER_IP>:5555 \
  --task="<TASK_INSTRUCTION>" \
  --actions_per_chunk=40 \
  --chunk_size_threshold=0.7 --debug_visualize_queue_size=True \
  --robot.type=rby1 --fps=<FPS>
```  
> [!NOTE]
> - When you use lerobot-robot-client, you should properly set the camera config with suitiable rotation option both for your robot's camera and the corresponding dataset
> - The ZMQ server `host` and `port` must match the robot client configuration.
> - <POLICY_SERVER_IP> : Replace it with the **actual IP address of
> the inference server** — do **not** use `0.0.0.0` from the server's startup log.