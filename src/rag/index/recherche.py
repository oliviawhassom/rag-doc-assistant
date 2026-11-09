"""Recherche des passages les plus proches d'une question (EF11).

La question reçoit le préfixe de requête (« query: » pour e5), est encodée par
le même modèle que les passages, puis comparée à l'index. Chaque passage
renvoyé porte son score (cosinus avec des vecteurs normalisés) et son rang.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.config import EmbeddingsConfig
from rag.index.encodeur import Encodeur
from rag.index.vectoriel import IndexVectoriel
from rag.models import Passage


@dataclass(frozen=True)
class ResultatRecherche:
    rang: int
    score: float
    passage: Passage


class MoteurRecherche:
    def __init__(
        self,
        index: IndexVectoriel,
        passages: list[Passage],
        encodeur: Encodeur,
        embeddings: EmbeddingsConfig,
    ) -> None:
        self._index = index
        self._passages = {p.passage_id: p for p in passages}
        manquants = set(index.passage_ids) - self._passages.keys()
        if manquants:
            raise ValueError(f"{len(manquants)} passage(s) de l'index introuvables")
        self._encodeur = encodeur
        self._prefixe = embeddings.prefixe_requete

    def rechercher(self, question: str, k: int) -> list[ResultatRecherche]:
        vecteur = self._encodeur.encoder([self._prefixe + question])[0]
        return [
            ResultatRecherche(rang, score, self._passages[pid])
            for rang, (pid, score) in enumerate(self._index.rechercher(vecteur, k), start=1)
        ]
