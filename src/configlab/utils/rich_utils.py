from lightning.pytorch.utilities.rank_zero import rank_zero_only
from rich.console import Console


class DummyConsole:
    """A dummy console that does nothing, used for non-zero ranks in distributed training."""

    def __getattr__(self, name: str) -> lambda: None:
        """Mock any attribute access."""
        return lambda *args, **kwargs: None


def get_console() -> Console:
    """Get a rich console that only prints from the rank zero process in distributed training."""
    if rank_zero_only.rank == 0:
        return Console()
    return DummyConsole()


console = get_console()
