import argparse
import sys

from train import vggt_train, vggt_train_pairs


def main() -> None:
    """
    Entry point dispatcher. Use:
      python main.py multiview [args...]    -> runs train/vggt_train.py
      python main.py pairwise  [args...]    -> runs train/vggt_train_pairs.py
    """
    parser = argparse.ArgumentParser(
        description="CoVISION training entrypoint (multiview or pairwise)",
        add_help=False,  # defer help for sub-parsers
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["multiview", "pairwise"],
        help="Select training pipeline: multiview (graph-level) or pairwise.",
    )
    # Parse only the mode; pass the rest through
    args, remaining = parser.parse_known_args()

    # Rebuild argv for the downstream script so its own argparse sees the right args
    sys.argv = [sys.argv[0]] + remaining

    if args.mode == "multiview":
        vggt_train.main()
    else:
        vggt_train_pairs.main()


if __name__ == "__main__":
    main()
