import heapq
import itertools
from typing import TypeVar, Generic

REMOVED = "<removed-element>"

ELEMENT_TYPE = TypeVar("ELEMENT_TYPE")
PRIORITY_TYPE = TypeVar("PRIORITY_TYPE")


class Entry:
    __slots__ = ("priority", "count", "item")

    priority: PRIORITY_TYPE
    count: int
    item: ELEMENT_TYPE

    def __init__(self, priority, count, item):
        self.priority = priority
        self.count = count
        self.item = item

    @property
    def values(self) -> tuple[PRIORITY_TYPE, int, ELEMENT_TYPE]:
        return self.priority, self.count, self.item

    def __getitem__(self, index: int):
        return self.values[index]

    def __setitem__(self, key: int, value: PRIORITY_TYPE | int | ELEMENT_TYPE):
        if -3 < key < 0:
            key += 3

        match key:
            case 0:
                self.priority = value
            case 1:
                self.count = value
            case 2:
                self.item = value
            case _:
                raise IndexError("Entry has only 3 slots")

    def __gt__(self, other: "Entry"):
        return self.values > other.values

    def __ge__(self, other: "Entry"):
        return self.values >= other.values

    def __lt__(self, other: "Entry"):
        return self.values < other.values

    def __le__(self, other: "Entry"):
        return self.values <= other.values


class PriorityQueue(Generic[ELEMENT_TYPE, PRIORITY_TYPE]):
    def __init__(self):
        self.elements: list[Entry] = []
        self.entry_finder: dict[ELEMENT_TYPE, Entry] = {}
        self.counter = itertools.count()

    def __bool__(self):
        return bool(self.entry_finder)

    def add(self, element: ELEMENT_TYPE, priority: PRIORITY_TYPE):
        # assert element not in self.entry_finder, "Element must not be in the priority queue"
        count = next(self.counter)
        entry = Entry(priority, count, element)
        self.entry_finder[element] = entry
        heapq.heappush(self.elements, entry)

    def update(self, element: ELEMENT_TYPE, priority: PRIORITY_TYPE):
        # assert element in self.entry_finder, "Element must be in the priority queue"
        self.remove(element)
        self.add(element, priority)

    def remove(self, item):
        entry = self.entry_finder.pop(item)
        entry[-1] = REMOVED

    def pop(self) -> tuple[ELEMENT_TYPE, PRIORITY_TYPE]:
        priority: PRIORITY_TYPE
        count: int
        item: ELEMENT_TYPE

        while self.elements:
            priority, count, item = heapq.heappop(self.elements)
            if item is not REMOVED:
                del self.entry_finder[item]
                return item, priority
        raise KeyError("Pop from an empty priority queue")

    def __contains__(self, item: ELEMENT_TYPE):
        return item in self.entry_finder

    def get_element(self, item: ELEMENT_TYPE) -> ELEMENT_TYPE:
        return self.entry_finder[item].item

    def get_smallest(self) -> tuple[ELEMENT_TYPE, PRIORITY_TYPE]:
        while self.elements[0].item is REMOVED:
            heapq.heappop(self.elements)
        return self.elements[0].item, self.elements[0].priority


class DoublyPriorityQueue(Generic[ELEMENT_TYPE, PRIORITY_TYPE]):
    """
    Orders elements with two priorities separately
    """

    def __init__(self):
        self.elements_1: list[Entry] = []
        self.elements_2: list[Entry] = []
        self.entry_finder: dict[ELEMENT_TYPE, tuple[Entry, Entry]] = {}
        self.counter = itertools.count()

    def __bool__(self):
        return bool(self.entry_finder)

    def add(self, element: ELEMENT_TYPE, priority_1: PRIORITY_TYPE, priority_2: PRIORITY_TYPE):
        count = next(self.counter)
        entry_1 = Entry(priority_1, count, element)
        entry_2 = Entry(priority_2, count, element)
        self.entry_finder[element] = entry_1, entry_2
        heapq.heappush(self.elements_1, entry_1)
        heapq.heappush(self.elements_2, entry_2)

    def update(self, element: ELEMENT_TYPE, priority_1: PRIORITY_TYPE, priority_2: PRIORITY_TYPE):
        self.remove(element)
        self.add(element, priority_1, priority_2)

    def remove(self, item):
        entry_1, entry_2 = self.entry_finder.pop(item)
        entry_1[-1] = REMOVED
        entry_2[-1] = REMOVED

    def pop(self) -> tuple[ELEMENT_TYPE, PRIORITY_TYPE, PRIORITY_TYPE]:
        priority_1: PRIORITY_TYPE
        priority_2: PRIORITY_TYPE
        count: int
        item: ELEMENT_TYPE

        while self.elements_1:
            priority_1, count, item = heapq.heappop(self.elements_1)
            if item is not REMOVED:
                entry_2 = self.entry_finder[item][1]
                entry_2[-1] = REMOVED
                priority_2 = self.entry_finder[item][1].priority
                del self.entry_finder[item]
                return item, priority_1, priority_2
        raise KeyError("Pop from an empty priority queue")

    def __contains__(self, item: ELEMENT_TYPE):
        return item in self.entry_finder

    def get_element(self, item: ELEMENT_TYPE) -> ELEMENT_TYPE:
        return self.entry_finder[item][0].item

    def get_smallest(self, priority_key: int) -> tuple[ELEMENT_TYPE, PRIORITY_TYPE]:
        elements = self.elements_1 if priority_key == 0 else self.elements_2
        while elements[0].item is REMOVED:
            heapq.heappop(elements)
        return elements[0].item, elements[0].priority
