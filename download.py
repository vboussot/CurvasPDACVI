from huggingface_hub import hf_hub_download
import os
import shutil

files = [
    "FT_0.pt",
    "FT_1.pt",
    "FT_2.pt",
    "FT_3.pt",
    "FT_4.pt",
    "M291.pt",
]

for file in files:
    cached_path = hf_hub_download(repo_id="vboussot/Curvas2025", filename=file, repo_type="model")
    local_path = os.path.join("./resources/Model/", file)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    shutil.copy(cached_path, local_path)