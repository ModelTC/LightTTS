######## lora 语音速度测试 ###############
from datetime import datetime
import time
import requests
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tqdm import tqdm
import os
import argparse
import torch

# 解析命令行参数
parser = argparse.ArgumentParser(description="Zero-shot TTS speed test script")
parser.add_argument("--port", type=int, default=8080, help="Server port number (default: 8080)")
parser.add_argument(
    "--cosyvoice_version", type=int, choices=[2, 3], default=3, help="CosyVoice version: 2 or 3 (default: 3)"
)
args = parser.parse_args()

# config
server_url = f"http://localhost:{args.port}/inference_zero_shot"
num_workers = [1, 2, 4, 8]
num_test = 50
path = "../cosyvoice/asset/zero_shot_prompt.wav"
sample_rate = 24000
# Get device name for file naming
device_name = torch.cuda.get_device_name(0).replace(" ", "_").lower() if torch.cuda.is_available() else "cpu"
# get all inputs
with open("test_texts.json", "r") as f:
    all_inputs = json.load(f)

failed = 0
os.makedirs("./outs", exist_ok=True)


def get_file(index):
    url = f"{server_url}/"
    inputs = all_inputs[index % len(all_inputs)]
    files = {"prompt_wav": ("sample.wav", open(path, "rb"), "audio/wav")}

    # 根据 cosyvoice_version 设置 prompt_text
    if args.cosyvoice_version == 3:
        prompt_text = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
    else:
        prompt_text = "希望你以后能够做的比我还好呦。"

    data = {
        "tts_text": inputs,
        "prompt_text": prompt_text,
    }
    global failed

    start = time.time()
    response = requests.post(url, files=files, data=data, stream=True)

    audio_data = bytearray()
    rtf = 0
    cost_time = 0

    try:
        if response.status_code == 200:
            for chunk in response.iter_content(chunk_size=4096):  # 分批读取
                if chunk:
                    audio_data.extend(chunk)
            cost_time = time.time() - start
            speech_len = len(audio_data) / 2 / 24000
            rtf = cost_time / speech_len if speech_len > 0 else 0
        else:
            print(f"{index} {inputs} \n error:{response.text}")
            failed += 1
            cost_time = time.time() - start
    except Exception as e:
        print(f"An exception occurred\ntext: {inputs}\n{e}\n{response.text}")
        failed += 1
        cost_time = time.time() - start
    finally:
        return cost_time, rtf


results = []
for num_worker in num_workers:
    print(f"Concurrency {num_worker} test starts")

    last_failed = failed

    start_time = time.time()
    with ProcessPoolExecutor(max_workers=num_worker) as executor:
        all_results = list(tqdm(executor.map(get_file, range(num_test)), total=num_test, desc="running tests"))
    total_time = time.time() - start_time

    all_cost_times = [result[0] for result in all_results]
    all_rtf = [result[1] for result in all_results]

    result = {"num_workers": num_worker}
    # 计算分位数
    if failed - last_failed > 0:
        print(f"Failed {failed - last_failed}")
    for percentile in [50, 90, 99]:
        percentile_data = np.percentile(np.array(all_cost_times), percentile)
        result[f"cost time {percentile}%"] = round(percentile_data, 2)
    for percentile in [50, 90, 99]:
        percentile_data = np.percentile(np.array(all_rtf), percentile)
        result[f"rtf {percentile}%"] = round(percentile_data, 2)
    result["avg rtf"] = round(np.mean(np.array(all_rtf)), 2)
    result["total_cost_time"] = round(total_time, 2)
    result["qps"] = round(num_test / total_time, 2)
    results.append(result)

    for k, v in result.items():
        print(f"{k}: {v}")

    print(f"Concurrency {num_worker} test ends")
    print("\n" + "=" * 80 + "\n")

if failed == 0:
    print("Test passed!")
else:
    print(f"Total failed {failed}")

result_path = f"./summary_triton_flashdecoding_{device_name}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.md"

heads = results[0].keys()
with open(result_path, "w") as md_file:
    md_file.write("|")
    for head in heads:
        md_file.write(head + "|")
    md_file.write("\r\n")
    md_file.write("|")
    for _ in range(len(heads)):
        md_file.write("------|")
    md_file.write("\r\n")
    for result in results:
        md_file.write("|")
        for head in heads:
            md_file.write(str(result[head]) + "|")
        md_file.write("\r\n")
