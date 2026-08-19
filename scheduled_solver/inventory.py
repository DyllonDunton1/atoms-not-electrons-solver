"""Time-aware committed pallet inventory."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence

from .models import InventoryEvent, PalletSpec


class InventoryTimeline:
    def __init__(self, pallets: Sequence[PalletSpec]) -> None:
        self._capacity = {pallet.pallet_id: pallet.max_count for pallet in pallets}
        self._events: Dict[int, List[InventoryEvent]] = defaultdict(list)

    def stock_at(
        self,
        pallet_id: int,
        timestep: int,
        local_events: Iterable[InventoryEvent] = (),
    ) -> int:
        if pallet_id not in self._capacity:
            raise KeyError(pallet_id)
        stock = self._capacity[pallet_id]
        events = list(self._events.get(pallet_id, ())) + [
            event for event in local_events if event.pallet_id == pallet_id
        ]
        events.sort(key=lambda event: (event.timestep, 0 if event.kind == "pick" else 1))
        for event in events:
            if event.timestep >= timestep:
                break
            if event.kind == "pick":
                stock -= event.amount
            elif event.kind == "refill":
                stock = self._capacity[pallet_id]
            else:
                raise ValueError(f"Unknown inventory event {event.kind!r}")
            if stock < 0:
                raise ValueError(f"Pallet {pallet_id} stock became negative")
        return stock

    def events_are_feasible(self, additional_events: Iterable[InventoryEvent]) -> bool:
        additional = list(additional_events)
        affected = {event.pallet_id for event in additional}
        for pallet_id in affected:
            stock = self._capacity[pallet_id]
            events = list(self._events.get(pallet_id, ())) + [
                event for event in additional if event.pallet_id == pallet_id
            ]
            events.sort(key=lambda event: (event.timestep, 0 if event.kind == "pick" else 1))
            for event in events:
                if event.kind == "pick":
                    stock -= event.amount
                    if stock < 0:
                        return False
                elif event.kind == "refill":
                    stock = self._capacity[pallet_id]
                else:
                    raise ValueError(f"Unknown inventory event {event.kind!r}")
        return True

    def commit(self, events: Iterable[InventoryEvent]) -> None:
        event_list = list(events)
        if not self.events_are_feasible(event_list):
            raise ValueError("Inventory events would invalidate committed stock timeline")
        for event in event_list:
            if event.pallet_id not in self._capacity:
                raise KeyError(event.pallet_id)
            if event.kind not in {"pick", "refill"}:
                raise ValueError(f"Unknown inventory event {event.kind!r}")
            self._events[event.pallet_id].append(event)
            self._events[event.pallet_id].sort(
                key=lambda item: (item.timestep, 0 if item.kind == "pick" else 1)
            )

    def events_for(self, pallet_id: int) -> tuple:
        return tuple(self._events.get(pallet_id, ()))
