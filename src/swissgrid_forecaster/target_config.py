"""Bind an explicitly specified target to a source without inventing semantics."""
from dataclasses import dataclass
from .target_contract import TargetContract
from .source_registry import SourceRegistry
from .availability import nonempty


@dataclass(frozen=True, slots=True)
class TargetConfig:
    contract: TargetContract
    source_id: str
    source_record_id: str

    def __post_init__(self):
        if not isinstance(self.contract, TargetContract):
            raise ValueError('explicit TargetContract required')
        nonempty(self.source_id, 'source_id')
        nonempty(self.source_record_id, 'source_record_id')

    def validate(self, registry: SourceRegistry) -> None:
        source = registry.get(self.source_id)
        source.validate_unit(self.contract.unit)
        source.require_forecast_metadata()
        # A label need not be observable at forecast issue time. Source horizon
        # availability constrains input selection, not the future target label.
