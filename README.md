![Light TTS Banner](asset/light-tts.jpg)

# LightTTS

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://hub.docker.com/r/lighttts/light-tts)

**⚡ Lightning-Fast Text-to-Speech Inference & Service Framework**

**LightTTS** is a lightweight and high-performance text-to-speech (TTS) inference and service framework based on Python. It supports **CosyVoice2** and **CosyVoice3** models, built upon the [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) architecture and [LightLLM](https://github.com/ModelTC/lightllm) framework, with optimizations to support fast, scalable, and service-ready TTS deployment.

---

## ✨ Features

- 🚀 **Optimized LLM Inference**: The language model part of the TTS pipeline is accelerated using techniques from lightllm and supports high-throughput batch inference
- 🧩 **Shared Memory Timbre Manager with LRU**: Manages speaker/timbre embeddings in shared memory for fast access and minimal recomputation
- 🧱 **Modular Architecture (Encode–LLM–Decode)**: Refactored from LightLLM into three decoupled modules—Encoder, LLM, and Decoder—each running as separate processes for efficient task parallelism and scalability.
- 🌐 **Service Ready and Easy Integration**: Comes with an HTTP API for fast deployment and simple APIs for integration into other Python or web projects
- 🔄 **Bi-streaming Mode via WebSocket**: Supports interactive bi-directional streaming using WebSocket for low-latency, real-time TTS communication
- 🔄 **Multi-LLM Support**: Supports loading multiple llm weights from a single model directory
---

## ⚡️ Get Started

### Installation

- (Option 1 Recommended) Run with Docker
    ```bash
    # The easiest way to install LightTTS is by using the official image. You can directly pull and run the official image
    docker pull lighttts/light-tts:latest

    # Or you can manually build the image
    docker build -t light-tts:latest .

    # Run the image
    docker run -it --gpus all -p 8080:8080 --shm-size 4g -v your_local_path:/data/ light-tts:latest /bin/bash

    # (Recommended) Update to the latest LightTTS code
    git checkout main && git pull

    # Then proceed to model download and service startup (see below)

- (Option 2) Install from Source

    ```bash
    # Clone the repo
    git clone --recursive https://github.com/ModelTC/LightTTS.git
    cd LightTTS
    # If you failed to clone the submodule due to network failures, please run the following command until success
    # cd LightTTS
    # git submodule update --init --recursive

    # (Recommended) Create a new conda environment
    conda create -n light-tts python=3.10
    conda activate light-tts

    # Install dependencies (We use the latest torch==2.9.1, but other versions are also compatible)
    pip install -r requirements.txt

    # If you encounter sox compatibility issues
    # ubuntu
    sudo apt-get install sox libsox-dev
    # centos
    sudo yum install sox sox-devel
    ```

### Model Download

We now support CosyVoice2 and CosyVoice3 models.

```python
# ModelScope SDK model download (SDK模型下载)
from modelscope import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')

# For overseas users, HuggingFace SDK model download
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('FunAudioLLM/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
```

(We have already installed the ttsfrd package in the docker image. If you are using docker image, you can skip this installation)
For better text normalization performance, you can optionally install the ttsfrd package and unzip its resources. This step is not required — if skipped, the system will fall back to WeTextProcessing by default.

```bash
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
```

### Start the Model Service

**Note:** It is recommended to enable the `load_trt` parameter for acceleration. The default flow precision is fp16 for CosyVoice2 and fp32 for CosyVoice3.

**For CosyVoice2:**

```bash
python -m light_tts.server.api_server --model_dir ./pretrained_models/CosyVoice2-0.5B
```

**For CosyVoice3:**

```bash
python -m light_tts.server.api_server --model_dir ./pretrained_models/Fun-CosyVoice3-0.5B-2512
```

**With custom data type** (float32, bfloat16, or float16; default: float16):

```bash
# Use float32 for better accuracy or float16 for faster speed
python -m light_tts.server.api_server --model_dir ./pretrained_models/Fun-CosyVoice3-0.5B-2512 --data_type float32
```

**Available Parameters:**

The default values are usually the fastest and generally do not need to be adjusted. If you need to customize them, please refer to the following parameter descriptions:
- `load_trt`: Whether to load the flow_decoder in TensorRT mode (default: True).
- `data_type`: The data type for LLM inference (default: float16)
- `load_jit`: Whether to load the flow_encoder in JIT mode (default: False).
- `max_total_token_num`: LLM arg, total token count the GPU and model can support = `max_batch * (input_len + output_len)` (default: 64 * 1024)
- `max_req_total_len`: LLM arg, maximum value for `req_input_len + req_output_len` (default: 32768, matches `max_position_embeddings`)
- `graph_max_len_in_batch`: Maximum sequence length for CUDA graph capture in decoding stage (default: 32768)
- `graph_max_batch_size`: Maximum batch size for CUDA graph capture in decoding stage (default: 16)

For more parameters, see `light_tts/server/api_cli.py`

Wait for the service to initialize. The default address is `http://localhost:8080`.

### Multi-LLM Configuration

LightTTS supports loading multiple llm weights from a single model directory. You can configure multiple llm weights by creating a `config.json` file in your model directory.

**Example `config.json`:**

```json
{
    "lora_info": [
        {
            "style_name": "style0",
            "llm_path": "llm0.pt"
        },
        {
            "style_name": "style1",
            "llm_path": "llm1.pt"
        }
    ]
}
```

**Note:** 
- If `config.json` does not exist, the system will use a default single-style configuration
- The system will automatically start separate LLM processes for each style to support concurrent inference
- You can use the `tts_model_name` parameter to specify the style name when sending requests

### Request Examples

Once the service is running, you can interact with it through the HTTP API. We support three modes: **non-streaming**, **streaming**, and **bi-streaming**.

**Quick Start - Simple Request:**

```python
import requests
import numpy as np
import soundfile as sf

url = "http://localhost:8080/inference_zero_shot"

# For CosyVoice3, use special prompt format
prompt_text = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
# For CosyVoice2, use: prompt_text = "希望你以后能够做的比我还好呦。"

files = {"prompt_wav": open("your_prompt_audio.wav", "rb")}
data = {
    "tts_text": "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
    "prompt_text": prompt_text,
    "stream": False,  # Set to True for streaming mode
    "tts_model_name": "default",
}

response = requests.post(url, files=files, data=data, stream=True)

# Collect audio data
audio_data = bytearray()
for chunk in response.iter_content(chunk_size=4096):
    if chunk:
        audio_data.extend(chunk)

# Convert to numpy array and save as WAV
audio_np = np.frombuffer(audio_data, dtype=np.int16)
sf.write("output.wav", audio_np, samplerate=24000, subtype="PCM_16")
```

**More Examples:**

- **Non-streaming and Streaming**: Use `test/test_zero_shot.py` for examples, which prints metrics such as RTF (Real-Time Factor) and TTFT (Time To First Token)
- **Bi-streaming**: Uses WebSocket interface. See usage examples in `test/test_bistream.py`

## 📊 Performance Benchmarks

We have conducted performance benchmarks on different GPU configurations to demonstrate the throughput and latency characteristics of LightTTS in streaming mode. 

Model: `Fun-CosyVoice3-0.5B-2512` datatype: `float16`

### NVIDIA GeForce RTX 4090D
non-stream: `test/test_zs_speed.py`

|num_workers|cost time 50%|cost time 90%|cost time 99%|rtf 50%|rtf 90%|rtf 99%|avg rtf|total_cost_time|qps|
|------|------|------|------|------|------|------|------|------|------|
|1|0.61|1.09|1.51|0.13|0.16|0.22|0.13|33.95|1.47|
|2|0.8|1.24|1.71|0.15|0.22|0.25|0.16|21.46|2.33|
|4|1.02|1.88|2.27|0.22|0.29|0.38|0.23|15.31|3.27|
|8|1.76|2.36|3.48|0.33|0.49|0.62|0.36|12.18|4.1|

stream: `test/test_zs_stream.py`

|num_workers|cost time 50%|cost time 90%|cost time 99%|ttft 50%|ttft 90%|ttft 99%|rtf 50%|rtf 90%|rtf 99%|avg rtf|total_cost_time|qps|
|------|------|------|------|------|------|------|------|------|------|------|------|------|
|1|1.01|2.15|2.82|0.33|0.34|0.9|0.21|0.25|0.34|0.22|60.13|0.83|
|2|1.83|3.56|5.16|0.93|1.53|2.3|0.34|0.63|0.81|0.4|52.47|0.95|
|4|3.43|5.76|7.31|2.62|4.37|5.8|0.7|1.28|2.16|0.81|48.74|1.03|
|8|7.27|10.01|10.45|6.4|8.55|9.03|1.28|2.67|3.66|1.57|47.37|1.06|

### NVIDIA GeForce RTX 5090
non-stream

|num_workers|cost time 50%|cost time 90%|cost time 99%|rtf 50%|rtf 90%|rtf 99%|avg rtf|total_cost_time|qps|
|------|------|------|------|------|------|------|------|------|------|
|1|0.51|0.81|1.61|0.11|0.13|0.23|0.11|28.9|1.73|
|2|0.64|1.1|1.48|0.13|0.16|0.26|0.13|17.54|2.85|
|4|0.87|1.28|1.68|0.17|0.23|0.36|0.18|11.45|4.37|
|8|1.32|1.86|2.14|0.25|0.4|0.6|0.29|8.97|5.57|

stream

|num_workers|cost time 50%|cost time 90%|cost time 99%|ttft 50%|ttft 90%|ttft 99%|rtf 50%|rtf 90%|rtf 99%|avg rtf|total_cost_time|qps|
|------|------|------|------|------|------|------|------|------|------|------|------|------|
|1|0.76|1.41|2.27|0.28|0.3|0.31|0.16|0.18|0.22|0.16|44.06|1.13|
|2|1.45|2.34|3.46|0.74|1.28|1.75|0.27|0.45|0.7|0.3|38.82|1.29|
|4|2.9|4.04|4.7|2.16|3.03|3.4|0.5|1.04|1.51|0.61|37.75|1.32|
|8|5.78|7.74|8.49|5.01|6.73|7.35|1.03|2.09|2.85|1.22|37.67|1.33|

**Metrics Explanation:**
- **num_workers**: Number of concurrent workers
- **cost time**: Total request processing time in seconds (50th/90th/99th percentile)
- **ttft**: Time to First Token in seconds (50th/90th/99th percentile)
- **rtf**: Real-Time Factor (50th/90th/99th percentile)
- **avg rtf**: Average Real-Time Factor
- **total_cost_time**: Total benchmark duration in seconds
- **qps**: Queries Per Second

## License

This repository is released under the [Apache-2.0](LICENSE) license.

### Third-Party Code Attribution

This project includes code from [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) (Copyright Alibaba, Inc. and its affiliates), which is also licensed under Apache-2.0. The CosyVoice code is located in the `cosyvoice/` directory and has been integrated and modified as part of LightTTS. See the [NOTICE](NOTICE) file for complete attribution details.