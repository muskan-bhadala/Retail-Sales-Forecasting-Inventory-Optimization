
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import calculate_inventory_policy


def test_inventory_policy_returns_required_fields():
    result = calculate_inventory_policy(
        forecast=[10, 12, 11, 10, 13, 12, 10],
        residual_std=2,
        on_hand=10,
        lead_time=3,
        service_level=0.95,
        annual_demand=4000,
        ordering_cost=50,
        unit_cost=20,
        holding_rate=0.20,
    )

    expected = {
        "mean_lead_time_demand",
        "safety_stock",
        "reorder_point",
        "eoq",
        "recommended_order_qty",
        "stock_status",
    }
    assert expected.issubset(result.keys())
    assert result["recommended_order_qty"] >= 0


def test_low_stock_triggers_reorder():
    result = calculate_inventory_policy(
        forecast=[10] * 14,
        residual_std=1,
        on_hand=0,
        lead_time=3,
        service_level=0.95,
        annual_demand=3650,
        ordering_cost=50,
        unit_cost=10,
        holding_rate=0.20,
    )
    assert result["stock_status"] == "REORDER"
    assert result["recommended_order_qty"] > 0


def test_high_stock_does_not_trigger_reorder():
    result = calculate_inventory_policy(
        forecast=[10] * 14,
        residual_std=1,
        on_hand=10000,
        lead_time=3,
        service_level=0.95,
        annual_demand=3650,
        ordering_cost=50,
        unit_cost=10,
        holding_rate=0.20,
    )
    assert result["stock_status"] == "SUFFICIENT"
    assert result["recommended_order_qty"] == 0


def test_invalid_service_level_is_rejected():
    with pytest.raises(ValueError):
        calculate_inventory_policy(
            forecast=[10] * 7,
            residual_std=1,
            on_hand=20,
            lead_time=2,
            service_level=1.0,
            annual_demand=3650,
            ordering_cost=50,
            unit_cost=10,
            holding_rate=0.20,
        )
