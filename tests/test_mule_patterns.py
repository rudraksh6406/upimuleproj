"""Unit tests for all 9 mule pattern injection functions."""
import pytest
from src.data.mule_patterns import (
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


@pytest.fixture
def sample_accounts():
    return [
        {"account_id": f"acc_{i}", "vpa": f"user_{i}@okhdfcbank", "is_merchant": False, "is_mule": False}
        for i in range(30)
    ]


@pytest.fixture
def sample_merchants():
    return [
        {"account_id": f"m_{i}", "vpa": f"merchant.store{i}@okaxis", "is_merchant": True, "is_mule": False}
        for i in range(5)
    ]


def test_fan_out(sample_accounts, sample_merchants):
    source = sample_accounts[0]
    mules = sample_accounts[1:10]
    txns, flagged = inject_fan_out(source, mules, sample_merchants, start_time=1000.0)
    assert len(txns) >= 5
    assert len(flagged) >= 5


def test_layering_chain(sample_accounts):
    chain = sample_accounts[:4]
    txns, flagged = inject_layering_chain(chain, start_time=1000.0)
    assert len(txns) == 3
    assert len(flagged) == 4


def test_cyclic_flow(sample_accounts):
    cycle = sample_accounts[:3]
    txns, flagged = inject_cyclic_flow(cycle, start_time=1000.0)
    assert len(txns) == 3
    assert len(flagged) == 3


def test_burst_dormant(sample_accounts):
    dormant = sample_accounts[0]
    cps = sample_accounts[1:5]
    txns, flagged = inject_burst_dormant(dormant, cps, start_time=1000.0)
    assert len(txns) >= 5
    assert dormant["vpa"] in flagged


def test_device_sharing(sample_accounts):
    mules = sample_accounts[:10]
    recs = sample_accounts[10:15]
    txns, flagged = inject_device_sharing(mules, recs, start_time=1000.0)
    assert len(txns) >= 5
    # All transactions share the same device_id
    dev_ids = {t["device_id"] for t in txns}
    assert len(dev_ids) == 1
