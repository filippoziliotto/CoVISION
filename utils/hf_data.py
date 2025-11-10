from huggingface_hub import snapshot_download
from huggingface_hub import login

# login()
login(token="")
snapshot_download(
    repo_id="ai4ce/CoVISION_Reasoning",
    local_dir="data/",
    revision="main",
    repo_type="dataset",
)