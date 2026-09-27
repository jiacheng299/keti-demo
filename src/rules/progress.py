"""Serializable explanations of the state that actually drives each rule."""

from dataclasses import asdict, dataclass


@dataclass
class RuleProgress:
    rule_id: str
    label: str
    reason: str
    track_id: int | None = None
    current: float = 0
    required: float = 1
    unit: str = ""
    status: str = "waiting"

    def as_dict(self):
        return asdict(self)

    @property
    def text(self) -> str:
        identity = f" ID:{self.track_id}" if self.track_id is not None else ""
        count = f" {self.current:.1f}/{self.required:g}{self.unit}" if self.unit else ""
        return f"{self.label}{identity}{count} | {self.reason}"
