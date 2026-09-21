#!/usr/bin/env python3
"""CLI Script to generate synthetic UPI transactions and mule networks."""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.data.generator import UPIDataGenerator
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger("generate_data")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic UPI dataset with mule rings.")
    parser.add_argument("--config", type=str, default="config/data_config.yaml", help="Data config file path")
    parser.add_argument("--accounts_train", type=int, default=2000, help="Train accounts count")
    parser.add_argument("--accounts_val", type=int, default=1000, help="Val accounts count")
    parser.add_argument("--accounts_test", type=int, default=1000, help="Test accounts count")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config("data_config.yaml")
    generator = UPIDataGenerator(config=cfg)

    out_base = Path(args.output_dir)

    # 1. Training Set
    logger.info(f"Generating Training Set ({args.accounts_train} accounts, 14 days)...")
    txns_train, accs_train, mules_train = generator.generate_dataset(
        num_accounts=args.accounts_train, time_span_days=14, base_timestamp=1726910400.0
    )
    generator.save_dataset_to_disk(txns_train, accs_train, mules_train, str(out_base / "train"))
    logger.info(f"Train generated: {len(txns_train)} txns, {len(mules_train)} mules planted.")

    # 2. Validation Set
    logger.info(f"Generating Validation Set ({args.accounts_val} accounts, 5 days)...")
    txns_val, accs_val, mules_val = generator.generate_dataset(
        num_accounts=args.accounts_val, time_span_days=5, base_timestamp=1728120000.0
    )
    generator.save_dataset_to_disk(txns_val, accs_val, mules_val, str(out_base / "val"))
    logger.info(f"Val generated: {len(txns_val)} txns, {len(mules_val)} mules planted.")

    # 3. Test Set
    logger.info(f"Generating Test Set ({args.accounts_test} accounts, 5 days)...")
    txns_test, accs_test, mules_test = generator.generate_dataset(
        num_accounts=args.accounts_test, time_span_days=5, base_timestamp=1728552000.0
    )
    generator.save_dataset_to_disk(txns_test, accs_test, mules_test, str(out_base / "test"))
    logger.info(f"Test generated: {len(txns_test)} txns, {len(mules_test)} mules planted.")

    logger.info("Dataset generation complete! Files saved in data/raw/.")


if __name__ == "__main__":
    main()
