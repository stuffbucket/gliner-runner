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


class ModelDownloadRequiredError(ModelIntegrityError):
    """A known model is not installed in its configured provider location."""

    def __init__(self, model: str, revision: str, destination: str) -> None:
        self.model = model
        self.revision = revision
        self.destination = destination
        super().__init__(
            f"model {model!r} at revision {revision} is not installed; "
            "explicit download approval is required"
        )


class ModelMemoryLimitError(GlinerRunnerError):
    """A model cannot fit within the configured memory pressure limit."""

    def __init__(
        self,
        *,
        model: str,
        profile: str,
        limit_bytes: int,
        required_bytes: int,
        observed_bytes: int | None = None,
    ) -> None:
        self.model = model
        self.profile = profile
        self.limit_bytes = limit_bytes
        self.required_bytes = required_bytes
        self.observed_bytes = observed_bytes
        measurement = (
            f"observed {observed_bytes} bytes after load"
            if observed_bytes is not None
            else f"estimated minimum is {required_bytes} bytes"
        )
        super().__init__(
            f"model {model!r} with profile {profile} exceeds the configured "
            f"memory limit of {limit_bytes} bytes: {measurement}; no fallback was attempted"
        )
