"""Unit tests for synthetic UPI data generator and VPA builder."""
import pytest
from src.data.generator import UPIDataGenerator
from src.data.vpa_generator import generate_vpa, generate_device_id, sample_bank_handle


def test_vpa_generation():
    vpa_user = generate_vpa(is_merchant=False)
    assert "@" in vpa_user
    assert "." in vpa_user or "_" in vpa_user or any(c.isdigit() for c in vpa_user)

    vpa_merchant = generate_vpa(is_merchant=True)
    assert "@" in vpa_merchant
    assert "merchant." in vpa_merchant


def test_device_id_generation():
    dev1 = generate_device_id("test1")
    dev2 = generate_device_id("test2")
    assert dev1.startswith("dev_")
    assert dev1 != dev2


def test_dataset_generation():
    generator = UPIDataGenerator()
    txns, accs, mules = generator.generate_dataset(num_accounts=50, time_span_days=2)
    
    assert len(accs) == 50
    assert len(txns) > 0
    assert len(mules) > 0
    
    # Check temporal ordering
    timestamps = [t["timestamp"] for t in txns]
    assert timestamps == sorted(timestamps)
