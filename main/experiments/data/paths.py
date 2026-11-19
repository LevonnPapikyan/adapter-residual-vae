"""Where things are, relative to ``main/``.

What belongs here
-----------------
- ``DataPaths`` (file names inside a data folder) and the folders this code must
  never write to.

What does NOT belong here
-------------------------
- Reading any of the files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MAIN_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = MAIN_ROOT.parent
PROTECTED_DIRS = tuple(REPO_ROOT / name for name in ("models", "data", "figures"))  # read-only: released weights, data, reference results


@dataclass(frozen=True)
class DataPaths:
    """The files of one data folder."""

    data_dir: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir))

    @property
    def corpus(self) -> Path:
        """Cleaned 20 Newsgroups with the leak purged."""

        return self.data_dir / "corpus.json"

    @property
    def translations(self) -> Path:
        """The 1,000 test documents in five languages."""

        return self.data_dir / "translations.json"

    @property
    def octis_words(self) -> Path:
        """Cache of OCTIS's preprocessing."""

        return self.data_dir / "octis_words.json"

    @property
    def octis_dir(self) -> Path:
        """OCTIS's own preprocessed 20 Newsgroups."""

        return self.data_dir / "octis"

    @property
    def train_de(self) -> Path:
        """The training posts machine-translated into German."""

        return self.data_dir / "train_de.json"

    def teacher(self, name: str) -> Path:
        """A fitted LDA teacher."""

        return self.data_dir / name
