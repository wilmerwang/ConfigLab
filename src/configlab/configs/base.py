from dataclasses import dataclass


@dataclass
class TargetConfig:
    """Configuration for a target object that can be used for instantiation."""

    _target_: str
