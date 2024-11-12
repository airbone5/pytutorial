from huggingface_hub import snapshot_download
model_id="Qwen/Qwen2.5-Coder-7B-Instruct"
snapshot_download(repo_id=model_id, local_dir="../../pretrain/Qwen2.5-Coder-7B-Instruct",
                  local_dir_use_symlinks=False, revision="main")