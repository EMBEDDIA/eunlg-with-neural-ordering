import logging
from typing import List, Optional, Tuple

from core.document_planner import BodyDocumentPlanner, HeadlineDocumentPlanner
from core.models import Message

log = logging.getLogger("root")

MAX_PARAGRAPHS = 10

MAX_SATELLITES_PER_NUCLEUS = 0
MIN_SATELLITES_PER_NUCLEUS = 0

NEW_PARAGRAPH_ABSOLUTE_THRESHOLD = 0.0

SATELLITE_RELATIVE_THRESHOLD = 0.0
SATELLITE_ABSOLUTE_THRESHOLD = 0.0


class EUListBodyDocumentPlanner(BodyDocumentPlanner):
    def __init__(self) -> None:
        super().__init__(new_paragraph_absolute_threshold=NEW_PARAGRAPH_ABSOLUTE_THRESHOLD)

    def select_next_nucleus(
        self, available_message: List[Message], selected_nuclei: List[Message]
    ) -> Tuple[Message, float]:
        return _select_next_nucleus(available_message, selected_nuclei)

    def new_paragraph_relative_threshold(self, selected_nuclei: List[Message]) -> float:
        return _new_paragraph_relative_threshold(selected_nuclei)

    def select_satellites_for_nucleus(
        self, nucleus: Message, available_core_messages: List[Message], available_expanded_messages: List[Message]
    ) -> List[Message]:
        return []


class EUListHeadlineDocumentPlanner(HeadlineDocumentPlanner):
    def select_next_nucleus(
        self, available_message: List[Message], selected_nuclei: List[Message]
    ) -> Tuple[Message, float]:
        return _select_next_nucleus(available_message, selected_nuclei)


def _select_next_nucleus(
    available_messages: List[Message], selected_nuclei: List[Message]
) -> Tuple[Optional[Message], float]:

    log.debug("Starting a new paragraph")
    available = available_messages[:]  # copy

    if len(selected_nuclei) >= MAX_PARAGRAPHS or not available_messages:
        log.debug("MAX_PARAGPAPHS reached, stopping")
        return None, 0

    available.sort(key=lambda message: message.score, reverse=True)
    next_nucleus = available[0]

    return next_nucleus, next_nucleus.score


def _new_paragraph_relative_threshold(selected_nuclei: List[Message]) -> float:
    return float("-inf")
