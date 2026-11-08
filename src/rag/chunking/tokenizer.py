"""Comptage et positions des tokens (décision D4).

Un « token » est celui du tokenizer XLM-RoBERTa, partagé par multilingual-e5 et
bge-m3 : une taille de 256 tokens désigne exactement le même texte pour les
deux modèles d'embeddings comparés.

On ne se sert du tokenizer que pour des POSITIONS : ``positions(texte)`` donne,
pour chaque token, ses indices de début et de fin dans le texte. Un passage est
alors une tranche du texte d'origine, coupée exactement entre deux tokens ; on
ne reconstruit jamais du texte à partir des tokens (ce qui altérerait espaces
et caractères).

Le texte compté est le contenu seul : les tokens spéciaux du modèle, le préfixe
« passage: » de e5 et le chemin de section ajouté en contexte n'en font pas
partie. Ils s'ajoutent au moment de l'encodage (module 3), d'où la comparaison
des modèles à 256 tokens, loin de la limite de 512 de e5.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from tokenizers import Tokenizer

from rag.config import TokenizerConfig

logger = logging.getLogger(__name__)

_HASH_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class TokenizerIndisponible(RuntimeError):
    pass


class TokenizerTexte:
    """Enveloppe d'un ``tokenizers.Tokenizer`` : positions et comptage, sans troncature."""

    def __init__(self, tokenizer: Tokenizer, modele: str, revision: str) -> None:
        # Certains tokenizer.json embarquent une troncature (ex. 512 tokens) :
        # une longue section serait comptée trop courte, sans erreur.
        tokenizer.no_truncation()
        tokenizer.no_padding()
        self._tokenizer = tokenizer
        self.modele = modele
        self.revision = revision

    def positions(self, texte: str) -> list[tuple[int, int]]:
        return list(self._tokenizer.encode(texte, add_special_tokens=False).offsets)

    def compter(self, texte: str) -> int:
        return len(self._tokenizer.encode(texte, add_special_tokens=False).ids)


def charger_tokenizer(cfg: TokenizerConfig) -> TokenizerTexte:
    """Télécharge (ou relit dans le cache local) le tokenizer.json du modèle configuré."""
    from huggingface_hub import hf_hub_download

    try:
        chemin = Path(hf_hub_download(cfg.modele, "tokenizer.json", revision=cfg.revision))
    except Exception as exc:
        raise TokenizerIndisponible(
            f"tokenizer de {cfg.modele} (révision {cfg.revision}) inaccessible : {exc}. "
            "Il faut un accès à huggingface.co au premier lancement ; "
            "ensuite le cache local suffit."
        ) from exc

    # Le cache Hugging Face range chaque fichier sous snapshots/<hash du commit>/.
    revision_reelle = chemin.parent.name
    if not _HASH_COMMIT.match(cfg.revision):
        logger.warning(
            "Révision du tokenizer non figée (« %s »). Révision téléchargée : %s. "
            "Recopiez-la dans decoupage.tokenizer.revision pour la reproductibilité.",
            cfg.revision,
            revision_reelle,
        )
    return TokenizerTexte(Tokenizer.from_file(str(chemin)), cfg.modele, revision_reelle)
