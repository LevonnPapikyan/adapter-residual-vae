"""Score 2: can a linear probe tell a text's language from its topic mixture?

What belongs here
-----------------
- The cross-validated probe. Chance is one over the number of languages; lower is better.

What does NOT belong here
-------------------------
- Producing the mixtures.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def language_score(*, thetas: Mapping[str, np.ndarray]) -> float:
    """5-fold accuracy, with all versions of a document in the same fold.

    ``thetas`` maps each language to ``[docs, num_topics]`` mixtures, the same
    documents in the same order in every language.
    """

    features = np.concatenate(list(thetas.values()))
    num_docs = len(features) // len(thetas)
    labels = np.repeat(np.arange(len(thetas)), num_docs)
    groups = np.tile(np.arange(num_docs), len(thetas))
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    return float(cross_val_score(probe, features, labels, groups=groups, cv=GroupKFold(5)).mean())
