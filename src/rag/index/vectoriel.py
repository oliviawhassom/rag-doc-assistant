"""Index vectoriel FAISS exact (EF09, EF11, décision D9).

``IndexFlatIP`` compare la question à chaque passage : produit scalaire, égal
au cosinus quand les vecteurs sont normalisés. Recherche exacte, donc mêmes
résultats à chaque exécution sur une même machine (ENF06) ; avec quelques
milliers de passages, elle reste instantanée. Un index approximatif (HNSW, IVF)
n'apporterait de la vitesse qu'au-delà de centaines de milliers de passages,
au prix d'une variabilité qui fausserait les comparaisons de configurations.

Ordre des résultats : score décroissant, puis ``passage_id`` croissant en cas
d'égalité exacte (deux passages identiques, par exemple). Sans ce départage,
FAISS rendrait les ex æquo dans un ordre non garanti.

Enregistrement : l'index est sérialisé en mémoire puis écrit par Python, au
lieu de ``faiss.write_index``, qui échoue sous Windows sur les chemins
contenant des caractères non ASCII (un nom d'utilisateur accentué, par exemple).
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from rag.index.encodeur import Vecteurs

FICHIER_INDEX = "index.faiss"
FICHIER_IDS = "passage_ids.json"


class IndexVectoriel:
    def __init__(self, index: faiss.Index, passage_ids: list[str]) -> None:
        if index.ntotal != len(passage_ids):
            raise ValueError(
                f"index incohérent : {index.ntotal} vecteurs pour {len(passage_ids)} passages"
            )
        self._index = index
        self.passage_ids = passage_ids

    @classmethod
    def construire(cls, passage_ids: list[str], vecteurs: Vecteurs) -> IndexVectoriel:
        if vecteurs.ndim != 2 or vecteurs.shape[0] != len(passage_ids):
            raise ValueError(f"vecteurs de forme {vecteurs.shape} pour {len(passage_ids)} passages")
        index = faiss.IndexFlatIP(int(vecteurs.shape[1]))
        if len(passage_ids):
            index.add(np.ascontiguousarray(vecteurs, dtype=np.float32))
        return cls(index, list(passage_ids))

    @property
    def dimension(self) -> int:
        return int(self._index.d)

    def __len__(self) -> int:
        return int(self._index.ntotal)

    def rechercher(self, requete: Vecteurs, k: int) -> list[tuple[str, float]]:
        """Les ``k`` passages les plus proches : (passage_id, score), score décroissant."""
        n = len(self)
        if n == 0 or k <= 0:
            return []
        q = np.ascontiguousarray(requete.reshape(1, -1), dtype=np.float32)
        # On demande tous les scores (quelques milliers : instantané avec un index
        # exact) pour que le départage des ex æquo soit exact, même au rang k.
        scores, indices = self._index.search(q, n)
        resultats = [
            (self.passage_ids[int(i)], float(s))
            for s, i in zip(scores[0], indices[0], strict=True)
            if i >= 0
        ]
        resultats.sort(key=lambda r: (-r[1], r[0]))
        return resultats[:k]

    def sauvegarder(self, dossier: Path) -> None:
        dossier.mkdir(parents=True, exist_ok=True)
        (dossier / FICHIER_INDEX).write_bytes(faiss.serialize_index(self._index).tobytes())
        (dossier / FICHIER_IDS).write_text(json.dumps(self.passage_ids), encoding="utf-8")

    @classmethod
    def charger(cls, dossier: Path) -> IndexVectoriel:
        brut = np.frombuffer((dossier / FICHIER_INDEX).read_bytes(), dtype=np.uint8)
        ids = json.loads((dossier / FICHIER_IDS).read_text(encoding="utf-8"))
        return cls(faiss.deserialize_index(brut), ids)
