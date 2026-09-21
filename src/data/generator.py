"""Synthetic UPI Transaction Generator for MuleGuard.

Generates realistic UPI transactions adhering to the statistical distributions in PRD Section 6:
- Legitimate accounts with small-world counterpart networks
- Daily Poisson transaction frequencies and LogNormal amount distribution
- Injected mule rings across 9 topology patterns
- Strict temporal sorting and time-based train/val/test splits
"""
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .mule_patterns import (
    _create_txn_event,
    inject_amount_clustering,
    inject_burst_dormant,
    inject_coordinated_ring,
    inject_cyclic_flow,
    inject_device_sharing,
    inject_fan_in,
    inject_fan_out,
    inject_gaming_betting,
    inject_layering_chain,
)
from .vpa_generator import (
    INDIAN_CITIES,
    MERCHANT_CATEGORIES,
    UPI_APPS,
    generate_device_id,
    generate_ip_address,
    generate_vpa,
    sample_merchant_category,
)


class UPIDataGenerator:
    """Master UPI Transaction and Mule Network Generator."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.gen_cfg = self.config.get("generation", {})
        self.legit_cfg = self.config.get("legitimate", {})
        self.banks_cfg = self.config.get("banks", None)
        
        # Default distributions from PRD Section 6
        self.mule_ratio = self.gen_cfg.get("mule_ratio", 0.05)
        self.mu = self.legit_cfg.get("amount_lognormal_mu", 4.2)
        self.sigma = self.legit_cfg.get("amount_lognormal_sigma", 1.5)
        self.active_ratio = self.legit_cfg.get("active_ratio", 0.6)
        self.lambda_active = self.legit_cfg.get("daily_lambda_active", 5.0)
        self.lambda_casual = self.legit_cfg.get("daily_lambda_casual", 0.5)
        self.p2p_ratio = self.legit_cfg.get("p2p_ratio", 0.4)

    def generate_account_population(self, num_accounts: int) -> List[Dict[str, Any]]:
        """Generates a population of user and merchant accounts."""
        accounts = []
        num_merchants = max(5, int(num_accounts * 0.10))
        num_users = num_accounts - num_merchants

        # Generate Users
        for i in range(num_users):
            city, state = random.choice(INDIAN_CITIES)
            dev_count_r = random.random()
            if dev_count_r < 0.70:
                dev_count = 1
            elif dev_count_r < 0.95:
                dev_count = 2
            else:
                dev_count = 3

            primary_dev = generate_device_id()
            devices = [primary_dev] + [generate_device_id() for _ in range(dev_count - 1)]
            
            is_active = random.random() < self.active_ratio
            age_days = random.randint(1, 730)
            kyc = "full" if random.random() < 0.85 else "min"

            account = {
                "account_id": f"acc_{i:06d}",
                "vpa": generate_vpa(is_merchant=False, bank_weights=self.banks_cfg),
                "is_merchant": False,
                "is_active": is_active,
                "daily_lambda": self.lambda_active if is_active else self.lambda_casual,
                "account_age_days": age_days,
                "kyc_type": kyc,
                "primary_device": primary_dev,
                "devices": devices,
                "city": city,
                "state": state,
                "app_name": random.choice(UPI_APPS),
                "device_type": "android" if random.random() < 0.80 else "ios",
                "counterparts": [],
                "is_mule": False,
                "mule_pattern": None,
                "dormancy_days": 0.0,
            }
            accounts.append(account)

        # Generate Merchants
        for j in range(num_merchants):
            city, state = random.choice(INDIAN_CITIES)
            m_cat = sample_merchant_category()
            account = {
                "account_id": f"merc_{j:06d}",
                "vpa": generate_vpa(is_merchant=True, bank_weights=self.banks_cfg),
                "is_merchant": True,
                "merchant_category": m_cat,
                "is_active": True,
                "daily_lambda": 15.0,  # merchants receive high volume
                "account_age_days": random.randint(30, 730),
                "kyc_type": "full",
                "primary_device": generate_device_id(),
                "devices": [generate_device_id()],
                "city": city,
                "state": state,
                "app_name": "BHIM",
                "device_type": "android",
                "counterparts": [],
                "is_mule": False,
                "mule_pattern": None,
                "dormancy_days": 0.0,
            }
            accounts.append(account)

        # Build Social Network / Counterparts
        users = [a for a in accounts if not a["is_merchant"]]
        merchants = [a for a in accounts if a["is_merchant"]]

        for u in users:
            # 3 to 15 regular counterparts
            num_cp = random.randint(3, 15)
            # Pick friends/family (users) and favorite merchants
            user_cps = random.sample(users, min(len(users), max(2, num_cp - 2)))
            merc_cps = random.sample(merchants, min(len(merchants), 2))
            u["counterparts"] = [c["vpa"] for c in (user_cps + merc_cps) if c["vpa"] != u["vpa"]]

        return accounts

    def _sample_amount(self) -> float:
        """Samples transaction amount from LogNormal(mu, sigma)."""
        amt = np.random.lognormal(self.mu, self.sigma)
        return max(10.0, min(100000.0, float(amt)))

    def _sample_timestamp(self, day_offset: int, base_time: float) -> float:
        """Samples timestamp with bimodal daily activity peaks (10am-2pm, 6pm-10pm)."""
        r = random.random()
        if r < 0.45:
            # Peak 1: 10:00 to 14:00 (10 to 14 hrs)
            hour = random.uniform(10.0, 14.0)
        elif r < 0.85:
            # Peak 2: 18:00 to 22:00 (18 to 22 hrs)
            hour = random.uniform(18.0, 22.0)
        elif r < 0.95:
            # Afternoon / Morning: 6:00 to 10:00 or 14:00 to 18:00
            hour = random.choice([random.uniform(6.0, 10.0), random.uniform(14.0, 18.0)])
        else:
            # Late night: 22:00 to 06:00
            hour = random.choice([random.uniform(22.0, 24.0), random.uniform(0.0, 6.0)])

        seconds_in_day = hour * 3600.0 + random.uniform(0.0, 59.0)
        return base_time + (day_offset * 86400.0) + seconds_in_day

    def generate_dataset(
        self,
        num_accounts: int = 2000,
        time_span_days: int = 14,
        base_timestamp: float = 1726910400.0,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Set[str]]:
        """Generates a complete dataset with legitimate transactions and injected mule rings."""
        accounts = self.generate_account_population(num_accounts)
        users = [a for a in accounts if not a["is_merchant"]]
        merchants = [a for a in accounts if a["is_merchant"]]
        
        acc_by_vpa = {a["vpa"]: a for a in accounts}
        all_txns: List[Dict[str, Any]] = []
        all_mule_vpas: Set[str] = set()

        # 1. Generate Legitimate Background Transactions
        for day in range(time_span_days):
            for u in users:
                num_txns = np.random.poisson(u["daily_lambda"])
                for _ in range(num_txns):
                    if not u["counterparts"]:
                        continue
                    rec_vpa = random.choice(u["counterparts"])
                    receiver = acc_by_vpa.get(rec_vpa)
                    if not receiver:
                        continue
                    t = self._sample_timestamp(day, base_timestamp)
                    amt = self._sample_amount()
                    
                    is_p2m = receiver.get("is_merchant", False)
                    m_cat = receiver.get("merchant_category", None) if is_p2m else None
                    dev = u["primary_device"] if random.random() < 0.95 else random.choice(u["devices"])

                    txn = _create_txn_event(
                        txn_id=f"txn_legit_{int(t)}_{random.randint(1000, 99999)}",
                        sender=u,
                        receiver=receiver,
                        amount=amt,
                        timestamp=t,
                        txn_type="P2M" if is_p2m else "P2P",
                        device_id=dev,
                        merchant_category=m_cat,
                        is_mule_sender=False,
                        is_mule_receiver=False,
                        pattern_type=None,
                    )
                    all_txns.append(txn)

        # 2. Plant Mule Patterns
        num_target_mules = max(10, int(num_accounts * self.mule_ratio))
        # Select candidates for mule roles
        candidate_pool = [u for u in users if len(u["counterparts"]) >= 2]
        random.shuffle(candidate_pool)
        
        # Partition candidates into pattern groups
        mule_batches = [
            candidate_pool[i:i + 15] for i in range(0, min(len(candidate_pool), num_target_mules * 2), 15)
        ]

        pattern_injectors = [
            ("fan_out", lambda b, t: inject_fan_out(random.choice(users), b, merchants, t)),
            ("layering_chain", lambda b, t: inject_layering_chain(b, t, random.uniform(30000, 80000))),
            ("fan_in", lambda b, t: inject_fan_in(b, random.choice(users), t)),
            ("cyclic_flow", lambda b, t: inject_cyclic_flow(b[:4], t, random.uniform(20000, 50000))),
            ("burst_dormant", lambda b, t: inject_burst_dormant(b[0], users[:5], t)),
            ("coordinated_ring", lambda b, t: inject_coordinated_ring(b, random.choice(users), random.choice(users), t)),
            ("device_sharing", lambda b, t: inject_device_sharing(b, users[:5], t)),
            ("amount_clustering", lambda b, t: inject_amount_clustering(b, merchants, t)),
            ("gaming_betting", lambda b, t: inject_gaming_betting(b[:5], users[:10], t)),
        ]

        # Distribute injections over days
        for p_idx, (p_name, injector) in enumerate(pattern_injectors):
            if p_idx < len(mule_batches) and len(mule_batches[p_idx]) >= 3:
                day_for_pattern = random.randint(1, max(1, time_span_days - 2))
                start_t = base_timestamp + (day_for_pattern * 86400.0) + random.uniform(36000.0, 72000.0)
                
                injected_txns, mules_flagged = injector(mule_batches[p_idx], start_t)
                all_txns.extend(injected_txns)
                all_mule_vpas.update(mules_flagged)

        # 3. Sort strictly by timestamp (Crucial for Temporal Graph Networks)
        all_txns.sort(key=lambda x: x["timestamp"])

        return all_txns, accounts, all_mule_vpas

    def save_dataset_to_disk(
        self,
        txns: List[Dict[str, Any]],
        accounts: List[Dict[str, Any]],
        mule_vpas: Set[str],
        output_dir: str,
    ) -> None:
        """Saves transactions and labels to JSON / CSV."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Save Transactions
        txns_file = out_path / "transactions.json"
        with open(txns_file, "w", encoding="utf-8") as f:
            json.dump(txns, f, indent=2)

        # Save Accounts & Mule Labels
        accounts_file = out_path / "accounts.json"
        with open(accounts_file, "w", encoding="utf-8") as f:
            json.dump(accounts, f, indent=2)

        labels_file = out_path / "labels.json"
        with open(labels_file, "w", encoding="utf-8") as f:
            json.dump(list(mule_vpas), f, indent=2)
