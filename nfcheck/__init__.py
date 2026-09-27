"""nfcheck: prove or disprove "no functional change" claims in git history."""

__version__ = "0.1.0"

from .check import Checker, Claim, CommitResult, Verdict, claims_no_change  # noqa: E402
from .fingerprint import Fingerprint, Unparseable, fingerprint  # noqa: E402

__all__ = ["Checker", "Claim", "CommitResult", "Fingerprint", "Unparseable", "Verdict",
           "claims_no_change", "fingerprint", "__version__"]
