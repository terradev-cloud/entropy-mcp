"""Error types for entropy-mcp.

Every error raised inside a tool handler is converted to a structured
dict by the dispatch layer:

    {
        "error": {
            "code":      machine-readable string,
            "message":   human-readable description,
            "parameter": the offending parameter name (or null),
            "expected":  the valid range or expected format (or null),
        }
    }

Agents recover from errors far better when the error names the fix, so
every raise site should fill in `parameter` and `expected` when known.
"""


class EntropyError(Exception):
    """Base class for all entropy-mcp errors."""

    code = "entropy_error"

    def __init__(self, message, parameter=None, expected=None):
        super().__init__(message)
        self.message = message
        self.parameter = parameter
        self.expected = expected

    def to_dict(self):
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "parameter": self.parameter,
                "expected": self.expected,
            }
        }


class ParameterError(EntropyError):
    """A parameter was missing, out of range, or malformed."""

    code = "parameter_error"


class SeedError(EntropyError):
    """A seed was malformed, or a revealed seed does not match."""

    code = "seed_error"


class CommitmentNotFound(EntropyError):
    """No commitment with the given id exists in the local log."""

    code = "commitment_not_found"


class VerificationError(EntropyError):
    """A verification step failed in a way that is not a clean false."""

    code = "verification_error"


class StampUnavailable(EntropyError):
    """Stamp is unreachable and the requested operation requires it."""

    code = "stamp_unavailable"


class BeaconUnavailable(EntropyError):
    """The drand beacon is unreachable (non-fatal where documented)."""

    code = "beacon_unavailable"
