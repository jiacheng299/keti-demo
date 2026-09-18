"""Stateful border-scene rules and orchestration."""

from .base_rule import EventCandidate, FrameState
from .region_rules import EnterRegionRule, LeaveRegionRule
from .rule_engine import RuleEngine
from .temporal_rules import DwellRule

__all__ = [
    "DwellRule",
    "EnterRegionRule",
    "EventCandidate",
    "FrameState",
    "LeaveRegionRule",
    "RuleEngine",
]
