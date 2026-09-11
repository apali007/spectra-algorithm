"""Check output-length distribution by case_label for a reasoning-corpus
dataset, to investigate whether certain case types (e.g. longer, more
complex reasoning traces) correlate with training instability."""
import argparse
import json
import logging
from collections import defaultdict, Counter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)


def build_argparser():
    parser = argparse.ArgumentParser(description="Output-length distribution by case_label")
    parser.add_argument("--input", type=str, required=True, help="Path to a train/val/test JSON")
    parser.add_argument("--top_n", type=int, default=10,
                         help="How many of the longest examples to print details for")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} examples from {args.input}")

    # Char-length is a fast proxy — good enough to compare relative
    # complexity across case types without loading a tokenizer.
    by_case = defaultdict(list)
    for item in data:
        by_case[item.get('case_label', 'UNKNOWN')].append((len(item.get('output', '')), item))

    logger.info("Output length by case_label:")
    for case, entries in sorted(by_case.items()):
        lens = sorted(e[0] for e in entries)
        n = len(lens)
        logger.info(f"  {case} (n={n}): min={lens[0]}, max={lens[-1]}, "
                    f"mean={sum(lens)/n:.0f}, p95={lens[int(n*0.95)] if n > 1 else lens[0]}")

    # Overall longest examples across the whole dataset, regardless of
    # case_label — these are the ones most likely to approach
    # max_seq_length or dominate a batch's compute/gradient behavior.
    all_entries = [(len(item.get('output', '')), item) for item in data]
    all_entries.sort(key=lambda x: x[0], reverse=True)

    logger.info(f"\nTop {args.top_n} longest examples overall:")
    for length, item in all_entries[:args.top_n]:
        logger.info(f"  len={length} | case={item.get('case_label')} | "
                    f"peptide={item.get('peptide')} | raw_file={item.get('raw_file')} | "
                    f"scan_number={item.get('scan_number')}")


if __name__ == "__main__":
    main()