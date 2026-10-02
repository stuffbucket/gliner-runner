class GlinerRunnerError(Exception):
    """Base exception for runner failures."""


class BackendUnavailableError(GlinerRunnerError):
    """The explicitly requested backend cannot be used."""


class UnsupportedCapabilityError(GlinerRunnerError):
    """A backend does not implement the requested operation or precision."""


class QueueFullError(GlinerRunnerError):
    """The bounded scheduler queue has reached capacity."""


class RunnerClosedError(GlinerRunnerError):
    """The scheduler is not accepting work."""


class ModelIntegrityError(GlinerRunnerError):
    """A model artifact failed manifest validation."""
