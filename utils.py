import os
import requests
from tqdm import tqdm

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def download_file(url, local_path):
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(local_path, 'wb') as f, tqdm(desc=local_path, total=total, unit='iB', unit_scale=True) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def format_chat_prompt(instruction, response=None):
    if response is None:
        return f"### User: {instruction}\n### Assistant: "
    else:
        return f"### User: {instruction}\n### Assistant: {response}</s>"