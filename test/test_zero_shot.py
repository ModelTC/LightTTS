import requests
import time
import soundfile as sf
import numpy as np
import os
import threading
import json
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser(description="Zero-shot TTS test script")
parser.add_argument("--port", type=int, default=8080, help="Server port number (default: 8080)")
parser.add_argument(
    "--cosyvoice_version", type=int, choices=[2, 3], default=3, help="CosyVoice version: 2 or 3 (default: 3)"
)
parser.add_argument("--stream", action="store_true", default=False, help="是否使用流式推理 (default: True)")
parser.add_argument("--num", type=int, default=5, help="测试数量 (default: 5)")
args = parser.parse_args()

url = f"http://0.0.0.0:{args.port}/inference_zero_shot"
num = args.num
# 准备要发送的文本和音频文件
path = "../cosyvoice/asset/zero_shot_prompt.wav"
stream = args.stream  # 是否使用流式推理
with open("test_texts.json", "r") as f:
    all_inputs = json.load(f)
res_list = []
os.makedirs("./outs", exist_ok=True)


def get_file(index):
    files = {"prompt_wav": ("sample.wav", open(path, "rb"), "audio/wav")}
    # inputs = random.choice(all_inputs)
    inputs = all_inputs[0]
    # inputs = all_inputs[2]

    # 根据 cosyvoice_version 设置 prompt_text
    if args.cosyvoice_version == 3:
        prompt_text = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
    else:
        prompt_text = "希望你以后能够做的比我还好呦。"

    data = {"tts_text": inputs, "prompt_text": prompt_text, "stream": stream}
    start_time = time.time()

    response = requests.post(url, files=files, data=data, stream=True)
    sample_rate = 24000
    first = True
    ttft = 0
    audio_data = bytearray()
    try:
        for chunk in response.iter_content(chunk_size=4096):  # 分批读取
            # if a_time > 0.01:
            if chunk:
                if first:
                    first = False
                    ttft = time.time() - start_time
                # print(f"Received {len(chunk)} bytes, Cost {(time.time() - chunk_time) * 1000} ms")
                audio_data.extend(chunk)
    except Exception as e:
        print(f"Exception: {e}")
        print(f"Error: {response.status_code}, {response.text}")
        return

    cost_time = time.time() - start_time
    speech_len = len(audio_data) / 2 / 24000
    # 将字节数据转换为 NumPy 数组
    audio_np = np.frombuffer(audio_data, dtype=np.int16)

    # 输出服务器响应
    if response.status_code == 200:
        # with open("output_audio.wav", "wb") as f:
        #     f.write(response.content)
        output_wav = f"./outs/output{'_stream' if stream else ''}_{index}.wav"
        sf.write(output_wav, audio_np, samplerate=sample_rate, subtype="PCM_16")
        print(
            f"{inputs} saved as {output_wav}, time cost: {cost_time:.2f} s"
            + f", rtf: {cost_time / speech_len}, ttft: {ttft:.2f} s"
        )
    else:
        print("Error:", response.status_code, response.text)


st = time.time()
for index in range(num):
    print("start index", index)
    thread = threading.Thread(target=get_file, args=(index,))
    thread.start()
