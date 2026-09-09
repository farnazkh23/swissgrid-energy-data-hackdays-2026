"""Configuration-driven mapping from arbitrary tabular columns to canonical fields."""
from dataclasses import dataclass
from collections.abc import Mapping


CANONICAL_FIELDS = frozenset({
    "source_id", "country", "domain", "unit", "value", "event_time", "valid_time",
    "known_at", "first_received_at", "normalized_at", "publication_time", "revision_id",
    "revision_sequence", "supersedes_revision_id", "source_record_id", "issue_time", "horizon", "target_name",
    "target_entity", "sign_convention", "resolution",
})
_MISSING = object()


@dataclass(frozen=True, slots=True)
class ColumnMapping:
    """A mapping has no opinion about provider column names or target semantics."""
    columns: tuple[tuple[str, str], ...]
    constants: tuple[tuple[str, object], ...] = ()

    def __post_init__(self):
        columns = tuple(tuple(pair) for pair in self.columns)
        constants = tuple(tuple(pair) for pair in self.constants)
        for name, value in (*columns, *constants):
            if not isinstance(name, str) or not name.strip():
                raise ValueError("mapping field names must be nonempty")
            if name not in CANONICAL_FIELDS:
                raise ValueError(f"unknown canonical field: {name}")
        if any(not isinstance(value, str) or not value.strip() for _, value in columns):
            raise ValueError("mapped source columns must be nonempty strings")
        if len({value for _, value in columns}) != len(columns):
            raise ValueError("one source column cannot map to multiple canonical fields")
        if len({name for name, _ in columns}) != len(columns) or len({name for name, _ in constants}) != len(constants):
            raise ValueError("duplicate canonical mapping")
        if set(dict(columns)) & set(dict(constants)):
            raise ValueError("a field cannot be both column-mapped and constant")
        object.__setattr__(self, "columns", tuple(sorted(columns)))
        object.__setattr__(self, "constants", tuple(sorted(constants, key=lambda pair: pair[0])))

    @classmethod
    def from_dict(cls, payload: Mapping) -> "ColumnMapping":
        if not isinstance(payload, Mapping):
            raise ValueError("column mapping configuration must be an object")
        columns = payload.get("columns", {})
        constants = payload.get("constants", {})
        if not isinstance(columns, Mapping) or not isinstance(constants, Mapping):
            raise ValueError("columns and constants must be objects")
        return cls(tuple(columns.items()), tuple(constants.items()))

    def column_for(self, field: str) -> str | None:
        return dict(self.columns).get(field)

    def has(self, field: str) -> bool:
        return field in dict(self.columns) or field in dict(self.constants)

    def value(self, row: Mapping, field: str, *, default=_MISSING):
        if field in dict(self.constants):
            return dict(self.constants)[field]
        column = self.column_for(field)
        if column is not None:
            if column not in row:
                raise ValueError(f"mapped column is absent from dataset: {column}")
            value = row[column]
            if value is not None and value != "":
                return value
        if default is not _MISSING:
            return default
        raise ValueError(f"required mapped field is missing: {field}")

    def require(self, *fields: str) -> None:
        missing = [field for field in fields if not self.has(field)]
        if missing:
            raise ValueError("mapping is missing required fields: " + ", ".join(missing))

    def to_dict(self) -> dict:
        return {"columns": dict(self.columns), "constants": dict(self.constants)}
