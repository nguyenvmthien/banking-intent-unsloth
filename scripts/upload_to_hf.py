import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


REQUIRED_FILES = ("adapter_model.safetensors", "adapter_config.json")


def resolve_folder_path(folder_path):
    if os.path.isabs(folder_path):
        return folder_path
    return os.path.normpath(os.path.join(os.path.dirname(__file__), folder_path))


def contains_required_files(folder_path):
    return all((Path(folder_path) / filename).exists() for filename in REQUIRED_FILES)


def find_checkpoint_folder(explicit_folder=None):
    if explicit_folder:
        resolved_folder = resolve_folder_path(explicit_folder)
        if not os.path.exists(resolved_folder):
            raise FileNotFoundError(f"Upload folder not found: {resolved_folder}")
        if not contains_required_files(resolved_folder):
            raise FileNotFoundError(
                f"Folder exists but is missing required model files: {resolved_folder}. "
                f"Expected: {', '.join(REQUIRED_FILES)}"
            )
        return resolved_folder

    candidate_roots = [
        resolve_folder_path("../outputs/checkpoint"),
        resolve_folder_path("../scripts/outputs"),
    ]

    candidate_folders = []
    for root in candidate_roots:
        if os.path.isdir(root):
            if contains_required_files(root):
                candidate_folders.append(root)
            else:
                for child_name in sorted(os.listdir(root), reverse=True):
                    child_path = os.path.join(root, child_name)
                    if os.path.isdir(child_path) and contains_required_files(child_path):
                        candidate_folders.append(child_path)

    if candidate_folders:
        return candidate_folders[0]

    raise FileNotFoundError(
        "No uploadable checkpoint found. Expected a folder containing adapter_model.safetensors and adapter_config.json. "
        "Train or restore the checkpoint first, then pass --folder <path> if needed."
    )


def main():
    parser = argparse.ArgumentParser(description="Upload a fine-tuned checkpoint folder to Hugging Face Hub")
    parser.add_argument("--repo-id", required=True, help="HF repo id in the form username/repo-name")
    parser.add_argument("--folder", default=None, help="Folder to upload; auto-detects a checkpoint if omitted")
    parser.add_argument("--private", action="store_true", help="Create the repo as private")
    parser.add_argument("--commit-message", default="Upload fine-tuned intent classifier")
    parser.add_argument("--token", default=None, help="HF token; otherwise reads HUGGINGFACE_HUB_TOKEN or HF_TOKEN")
    args = parser.parse_args()

    token = args.token or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    api = HfApi(token=token)

    folder_path = find_checkpoint_folder(args.folder)

    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=folder_path,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message,
    )


if __name__ == "__main__":
    main()