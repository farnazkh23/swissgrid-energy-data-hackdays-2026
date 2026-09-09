"""Explicit registry for configured sources; no implicit production defaults."""
from collections.abc import Iterable
from .source_contracts import Domain, SourceContract


class SourceRegistry:
    def __init__(self, sources: Iterable[SourceContract] = ()):
        self._sources: dict[str, SourceContract] = {}
        for source in sources:
            self.register(source)

    def register(self, source: SourceContract) -> None:
        if not isinstance(source, SourceContract):
            raise ValueError('expected SourceContract')
        if source.source_id in self._sources:
            raise ValueError('source_id already registered')
        self._sources[source.source_id] = source

    def get(self, source_id: str) -> SourceContract:
        try:
            return self._sources[source_id]
        except KeyError as exc:
            raise ValueError(f'unregistered source: {source_id}') from exc

    def list(self, domain: Domain | str | None = None) -> tuple[SourceContract, ...]:
        selected = None if domain is None else Domain(domain)
        return tuple(s for _, s in sorted(self._sources.items())
                     if selected is None or s.domain == selected)
