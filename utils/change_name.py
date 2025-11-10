#!/usr/bin/env python3
import os
from pathlib import Path


def rename_scene_folders(root: Path) -> None:
    """
    Under root (data/predictions_feat), rename any folder whose name does NOT
    contain '-' by appending '-1' to the name.
    """
    # Take a snapshot first so we don't confuse iteration while renaming
    subdirs = [p for p in root.iterdir() if p.is_dir()]

    for d in subdirs:
        name = d.name
        if "-" not in name:
            new_name = f"{name}-1"
            new_path = d.with_name(new_name)
            if new_path.exists():
                print(f"[WARN] Skipping rename of {d} -> {new_path} (target exists)")
                continue
            print(f"[INFO] Renaming folder: {d} -> {new_path}")
            d.rename(new_path)


def rename_embed_files(root: Path) -> None:
    """
    For each scene folder under root, find split_* / embs/ folders and
    rename any *.npy file whose name ends with 'last_embeds.npy' to
    exactly 'last_embeds.npy'.

    Example:
        0_last_embeds.npy -> last_embeds.npy
        3_last_embeds.npy -> last_embeds.npy
    """
    for scene_dir in root.iterdir():
        if not scene_dir.is_dir():
            continue

        for split_dir in scene_dir.glob("split_*"):
            if not split_dir.is_dir():
                continue

            embs_dir = split_dir / "embs"
            if not embs_dir.is_dir():
                continue

            for npy_path in embs_dir.glob("*.npy"):
                fname = npy_path.name
                # We want to collapse any prefix_*last_embeds.npy to last_embeds.npy
                if fname.endswith("last_embeds.npy") and fname != "last_embeds.npy":
                    target = npy_path.with_name("last_embeds.npy")
                    if target.exists():
                        print(f"[WARN] Target {target} already exists, skipping {npy_path}")
                        continue
                    print(f"[INFO] Renaming file: {npy_path} -> {target}")
                    npy_path.rename(target)


def main():
    root = Path("data/predictions_feat")

    if not root.exists():
        raise FileNotFoundError(f"Root folder not found: {root.resolve()}")

    # 1) Rename top-level scene folders (Adrian -> Adrian-1, etc.)
    rename_scene_folders(root)

    # 2) Rename embed files inside each scene/split_*/embs/ directory
    rename_embed_files(root)


if __name__ == "__main__":
    main()