"""Mock human escalation queue manager."""

from vay.types import HandoffTicket


class HandoffQueueManager:
    """Manages handoff queue tickets for live agent dashboard representation."""

    def __init__(self) -> None:
        self.tickets: list[HandoffTicket] = []

    def enqueue_ticket(self, ticket: HandoffTicket) -> None:
        """Enqueue escalation ticket for agent dashboard."""
        self.tickets.append(ticket)

    def get_pending_tickets(self) -> list[HandoffTicket]:
        """Retrieve all active pending tickets."""
        return self.tickets
