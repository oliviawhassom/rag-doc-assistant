"""Encodage des textes en vecteurs (EF08, décisions D4, D8, D11).

Le reste du code ne dépend que de l'interface ``Encodeur`` : changer de modèle
ne demande qu'une modification de configuration (critère d'EF08), et les tests
utilisent un encodeur factice.

``EncodeurST`` enveloppe sentence-transformers (D8). C'est le dépôt du modèle
sur Hugging Face qui décrit comment calculer le vecteur : moyenne des tokens
pour multilingual-e5, token initial pour bge-m3, puis normalisation. On ne le
réécrit pas : c'est l'endroit où une erreur passerait inaperçue.

Les préfixes (« query: », « passage: ») sont ajoutés par notre code, et
``prompt=""`` est passé explicitement à sentence-transformers : sans cela, il
appliquerait le préfixe par défaut déclaré par le modèle, s'il en déclare un,
et e5 recevrait « passage: passage: … » sans aucune erreur.

Chargement paresseux : la révision se résout sans charger les poids
(``resoudre_revision``). Quand l'index est à jour, le modèle n'est pas chargé.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from rag.config import EmbeddingsConfig

logger = logging.getLogger(__name__)

_HASH_COMMIT = re.compile(r"^[0-9a-f]{40}$")

Vecteurs = npt.NDArray[np.float32]


class ModeleIndisponible(RuntimeError):
    pass


class Encodeur(Protocol):
    modele: str
    revision: str
    dimension: int
    # Longueur maximale encodée par le modèle, tokens spéciaux compris.
    longueur_max: int

    def encoder(self, textes: list[str], progression: bool = False) -> Vecteurs: ...

    def compter_tokens(self, texte: str) -> int:
        """Longueur réellement encodée, tokens spéciaux compris (D11)."""
        ...


def resoudre_revision(modele: str, revision: str) -> str:
    """Hash du commit Hugging Face correspondant à ``revision``.

    Un hash déjà figé est renvoyé tel quel, sans réseau. « main » (ou une
    branche) est résolu via un petit fichier du dépôt, mis en cache, sans
    télécharger les poids ; un avertissement invite à figer la révision.
    """
    if _HASH_COMMIT.match(revision):
        return revision
    from huggingface_hub import hf_hub_download

    try:
        chemin = Path(hf_hub_download(modele, "config.json", revision=revision))
    except Exception as exc:
        raise ModeleIndisponible(
            f"modèle {modele} (révision {revision}) inaccessible : {exc}. "
            "Il faut un accès à huggingface.co au premier lancement ; "
            "ensuite le cache local suffit."
        ) from exc
    resolue = chemin.parent.name  # cache : snapshots/<hash du commit>/config.json
    logger.warning(
        "Révision du modèle %s non figée (« %s »). Révision utilisée : %s. "
        "Recopiez-la dans embeddings.revision pour la reproductibilité.",
        modele,
        revision,
        resolue,
    )
    return resolue


class EncodeurST:
    """Encodeur sentence-transformers, sur CPU (ENF04)."""

    def __init__(self, cfg: EmbeddingsConfig, revision: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dépend de l'installation
            raise ModeleIndisponible(
                "sentence-transformers n'est pas installé (pip install -r requirements.txt)"
            ) from exc
        try:
            self._st: Any = SentenceTransformer(cfg.modele, revision=revision, device="cpu")
        except Exception as exc:
            raise ModeleIndisponible(f"chargement de {cfg.modele} impossible : {exc}") from exc
        self.modele = cfg.modele
        self.revision = revision
        self.dimension = int(self._st.get_embedding_dimension() or 0)
        self.longueur_max = int(self._st.max_seq_length)
        self._normaliser = cfg.normaliser
        self._taille_lot = cfg.taille_lot

    def encoder(self, textes: list[str], progression: bool = False) -> Vecteurs:
        vecteurs = self._st.encode(
            textes,
            prompt="",  # aucun préfixe automatique : les nôtres sont déjà dans le texte
            batch_size=self._taille_lot,
            show_progress_bar=progression,
            convert_to_numpy=True,
            normalize_embeddings=self._normaliser,
        )
        return np.ascontiguousarray(vecteurs, dtype=np.float32)

    def compter_tokens(self, texte: str) -> int:
        ids = self._st.tokenizer(texte, add_special_tokens=True, truncation=False)["input_ids"]
        return len(ids)


def charger_encodeur(cfg: EmbeddingsConfig, revision: str) -> Encodeur:
    logger.info("Chargement du modèle %s (révision %s)", cfg.modele, revision)
    return EncodeurST(cfg, revision)
