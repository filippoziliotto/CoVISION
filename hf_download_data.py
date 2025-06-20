from huggingface_hub import snapshot_download
from huggingface_hub import login

# login()

snapshot_download(
    repo_id="ai4ce/CoVISION_Reasoning",
    local_dir="/path/to/directory/",
    revision="main",
    repo_type="dataset",
)