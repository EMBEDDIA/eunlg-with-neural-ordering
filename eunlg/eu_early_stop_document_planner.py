import logging
from typing import List, Optional, Tuple

from core.document_planner import BodyDocumentPlanner, HeadlineDocumentPlanner
from core.models import Message

log = logging.getLogger("root")

# MAX_PARAGRAPHS = 3
MAX_PARAGRAPHS = 2

MAX_SATELLITES_PER_NUCLEUS = 5
MIN_SATELLITES_PER_NUCLEUS = 2

NEW_PARAGRAPH_ABSOLUTE_THRESHOLD = 0.5

SATELLITE_RELATIVE_THRESHOLD = 0.5
SATELLITE_ABSOLUTE_THRESHOLD = 0.2


class EUEarlyStopBodyDocumentPlanner(BodyDocumentPlanner):
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
        return _select_satellites_for_nucleus(nucleus, available_core_messages, available_expanded_messages)


class EUEarlyStopHeadlineDocumentPlanner(HeadlineDocumentPlanner):
    def select_next_nucleus(
        self, available_message: List[Message], selected_nuclei: List[Message]
    ) -> Tuple[Message, float]:
        return _select_next_nucleus(available_message, selected_nuclei)


def _topic(message: Message) -> str:
    return ":".join(message.main_fact.value_type.split(":")[:3])


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
    # Gotta have something, so we'll add the first nucleus not matter value
    if not selected_nuclei:
        return float("-inf")

    # We'd really like to get a second paragraph, so we relax the requirements here
    if len(selected_nuclei) == 1:
        return 0

    # We already have at least 2 paragraphs, so we can be picky about whether we continue or not
    return 0.3 * selected_nuclei[0].score


def _select_satellites_for_nucleus(
    nucleus: Message, available_core_messages: List[Message], available_expanded_messages: List[Message]
) -> List[Message]:
    log.debug(
        "Selecting satellites for {} from among {} core messages and {} expanded messages".format(
            nucleus, len(available_core_messages), len(available_expanded_messages)
        )
    )
    satellites: List[Message] = []

    # Copy, s.t. we can modify in place
    available_core_messages = available_core_messages[:]
    available_expanded_messages = available_expanded_messages[:]

    while True:

        # Modify scores to account for context
        scored_available_core_messages = [
            (message.score, message) for message in available_core_messages if message.score > 0
        ]
        scored_available_expanded_messages = [
            (message.score, message) for message in available_expanded_messages if message.score > 0
        ]

        scored_available_messages = scored_available_core_messages + scored_available_expanded_messages

        # Filter out based on thresholds
        filtered_scored_available = [
            (score, message)
            for (score, message) in scored_available_messages
            if score > SATELLITE_RELATIVE_THRESHOLD * nucleus.score or score > SATELLITE_ABSOLUTE_THRESHOLD
        ]
        log.debug(
            "After rescoring for context, {} potential satellites remain".format(len(scored_available_core_messages))
        )

        if not filtered_scored_available:
            if len(satellites) >= MIN_SATELLITES_PER_NUCLEUS:
                log.debug("Done with satellites: MIN_SATELLITES_PER_NUCLEUS reached, no satellites pass filter.")
                return satellites
            elif scored_available_messages:
                log.debug(
                    "No satellite candidates pass threshold but have not reached MIN_SATELLITES_PER_NUCLEUS. "
                    "Trying without filter."
                )
                filtered_scored_available = scored_available_messages
            else:
                log.debug("Did not reach MIN_SATELLITES_PER_NUCLEUS, but ran out of candidates. Ending paragraphs.")
                return satellites

        if len(satellites) >= MAX_SATELLITES_PER_NUCLEUS:
            log.debug("Stopping due to having reaches MAX_SATELLITE_PER_NUCLEUS")
            return satellites

        filtered_scored_available.sort(key=lambda pair: pair[0], reverse=True)
        score, selected_satellite = filtered_scored_available[0]
        satellites.append(selected_satellite)
        log.debug("Added satellite {} (temp_score={})".format(selected_satellite, score))

        if selected_satellite in available_core_messages:
            available_core_messages = [message for message in available_core_messages if message != selected_satellite]
        else:
            available_expanded_messages = [
                message for message in available_expanded_messages if message != selected_satellite
            ]
