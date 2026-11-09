"""Construction ou rechargement de l'index (EF08, EF09, décisions D11, D12).

Un index est rechargé seulement si son empreinte est identique. L'empreinte
couvre tout ce qui détermine les vecteurs (D12) :

- les passages : identifiant et texte exactement tel qu'il est encodé, donc
  préfixe « passage: », chemin de section (``contexte_section``) et contenu ;
- le modèle et sa révision résolue (hash du commit, jamais « main ») ;
- toutes les options d'encodage : préfixes de requête et de passage,
  normalisation, chemin de section en tête.

Changer une seule de ces valeurs force la reconstruction, et la raison est
journalisée. Le dossier de l'index porte en plus une empreinte courte des
options : deux configurations d'expérience qui ne diffèrent que par une option
ont chacune leur index et ne s'écrasent pas.

``taille_lot`` n'entre pas dans l'empreinte : il ne règle que le nombre de
textes encodés à la fois. Les vecteurs peuvent toutefois différer très
légèrement d'une machine ou d'une version de torch à l'autre (calculs en
virgule flottante) : l'identité des résultats (ENF06) se vérifie sur une même
machine. Les versions des bibliothèques sont enregistrées dans les métadonnées.

Troncature (D11) : la taille d'un passage (EF06) ne compte que son contenu. La
longueur réellement encodée (préfixe, chemin, contenu, tokens spéciaux) est
vérifiée à part, contre la limite du modèle ; les passages qui la dépassent
seraient tronqués par le modèle, et ils sont comptés et signalés.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

import numpy as np

from rag.chunking import texte_a_encoder
from rag.config import Config, DecoupageConfig, EmbeddingsConfig
from rag.fichiers import ecrire_dossier_atomique
from rag.index.encodeur import Encodeur
from rag.index.vectoriel import IndexVectoriel
from rag.models import Passage

logger = logging.getLogger(__name__)

FICHIER_META = "meta.json"
_BIBLIOTHEQUES = ("torch", "sentence-transformers", "transformers", "faiss-cpu", "numpy")


@dataclass
class ResultatIndexation:
    index: IndexVectoriel
    dossier: Path
    statut: Literal["recharge", "construit"]
    raison: str
    meta: dict[str, Any]
    duree_s: float


def textes_encodes(
    passages: list[Passage], decoupage: DecoupageConfig, embeddings: EmbeddingsConfig
) -> list[str]:
    return [embeddings.prefixe_passage + texte_a_encoder(p, decoupage) for p in passages]


def parametres_encodage(
    decoupage: DecoupageConfig, embeddings: EmbeddingsConfig, revision: str
) -> dict[str, Any]:
    """Tout ce qui, en dehors des passages, détermine les vecteurs de l'index."""
    return {
        **embeddings.options_encodage(),
        "revision": revision,
        "contexte_section": decoupage.contexte_section,
        "config_decoupage": decoupage.identifiant,
    }


def calculer_empreinte(
    passages: list[Passage], textes: list[str], parametres: dict[str, Any]
) -> str:
    h = hashlib.sha256(json.dumps(parametres, sort_keys=True).encode())
    for p, t in zip(passages, textes, strict=True):
        h.update(p.passage_id.encode())
        h.update(b"\x00")
        h.update(t.encode())
        h.update(b"\x00")
    return h.hexdigest()


def dossier_index(
    dossier_store: Path, decoupage: DecoupageConfig, embeddings: EmbeddingsConfig
) -> Path:
    options = json.dumps(embeddings.options_encodage(), sort_keys=True).encode()
    modele = embeddings.modele.replace("/", "--")
    suffixe = hashlib.sha256(options).hexdigest()[:8]
    return dossier_store / "index" / f"{decoupage.identifiant}__{modele}-{suffixe}"


def lire_meta(dossier: Path) -> dict[str, Any] | None:
    chemin = dossier / FICHIER_META
    if not chemin.exists():
        return None
    contenu: dict[str, Any] = json.loads(chemin.read_text(encoding="utf-8"))
    return contenu


def _raison_reconstruction(ancien: dict[str, Any] | None, parametres: dict[str, Any]) -> str:
    if ancien is None:
        return "aucun index pour cette configuration"
    avant = ancien.get("parametres", {})
    changes = [
        f"{c} : {avant.get(c)!r} → {v!r}" for c, v in parametres.items() if avant.get(c) != v
    ]
    return "; ".join(changes) if changes else "passages modifiés"


def _versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for nom in _BIBLIOTHEQUES:
        try:
            versions[nom] = metadata.version(nom)
        except metadata.PackageNotFoundError:
            versions[nom] = None
    return versions


def indexer(
    passages: list[Passage],
    cfg: Config,
    revision: str,
    fabrique_encodeur: Callable[[], Encodeur],
    progression: bool = False,
) -> ResultatIndexation:
    """Recharge l'index si son empreinte est identique, sinon le reconstruit.

    ``fabrique_encodeur`` n'est appelée qu'en cas de reconstruction : un index
    à jour se recharge sans charger le modèle.
    """
    debut = time.perf_counter()
    parametres = parametres_encodage(cfg.decoupage, cfg.embeddings, revision)
    textes = textes_encodes(passages, cfg.decoupage, cfg.embeddings)
    empreinte = calculer_empreinte(passages, textes, parametres)
    dossier = dossier_index(cfg.stockage.dossier, cfg.decoupage, cfg.embeddings)
    ancien = lire_meta(dossier)

    if ancien is not None and ancien.get("empreinte") == empreinte:
        index = IndexVectoriel.charger(dossier)
        duree = time.perf_counter() - debut
        logger.info("Index rechargé depuis %s en %.2f s", dossier, duree)
        return ResultatIndexation(index, dossier, "recharge", "empreinte identique", ancien, duree)

    raison = _raison_reconstruction(ancien, parametres)
    logger.warning("Reconstruction de l'index %s : %s", dossier.name, raison)
    encodeur = fabrique_encodeur()
    if encodeur.revision != revision:
        raise ValueError(f"encodeur en révision {encodeur.revision}, attendu {revision}")

    longueurs = [encodeur.compter_tokens(t) for t in textes]
    tronques = [
        p.passage_id for p, n in zip(passages, longueurs, strict=True) if n > encodeur.longueur_max
    ]
    if tronques:
        logger.warning(
            "%d passage(s) dépassent la limite de %d tokens de %s et seront tronqués "
            "par le modèle : %s",
            len(tronques),
            encodeur.longueur_max,
            encodeur.modele,
            ", ".join(tronques[:10]) + (" …" if len(tronques) > 10 else ""),
        )

    debut_encodage = time.perf_counter()
    vecteurs = encodeur.encoder(textes, progression=progression)
    duree_encodage = time.perf_counter() - debut_encodage

    normes = np.linalg.norm(vecteurs, axis=1) if len(vecteurs) else np.zeros(0)
    if cfg.embeddings.normaliser and len(normes) and float(np.max(np.abs(normes - 1))) > 1e-3:
        raise ValueError("normalisation demandée mais des vecteurs n'ont pas une norme de 1")

    index = IndexVectoriel.construire([p.passage_id for p in passages], vecteurs)
    meta: dict[str, Any] = {
        "empreinte": empreinte,
        "parametres": parametres,
        "modele": encodeur.modele,
        "revision": encodeur.revision,
        "dimension": index.dimension,
        "longueur_max_modele": encodeur.longueur_max,
        "passages": len(passages),
        "passages_tronques": len(tronques),
        "exemples_tronques": tronques[:20],
        "longueur_encodee_max": max(longueurs, default=0),
        "duree_encodage_s": round(duree_encodage, 2),
        "passages_par_seconde": round(len(passages) / duree_encodage, 2)
        if duree_encodage
        else None,
        "date": datetime.now(UTC).isoformat(),
        "versions": _versions(),
    }

    def remplir(cible: Path) -> None:
        index.sauvegarder(cible)
        (cible / FICHIER_META).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    ecrire_dossier_atomique(dossier, remplir)
    duree = time.perf_counter() - debut
    logger.info(
        "Index construit : %d passages en %.1f s (%s passages/s)",
        len(passages),
        duree_encodage,
        meta["passages_par_seconde"],
    )
    return ResultatIndexation(index, dossier, "construit", raison, meta, duree)


def charger_index_a_jour(
    passages: list[Passage], cfg: Config, revision: str
) -> tuple[IndexVectoriel, dict[str, Any]]:
    """Recharge l'index pour la recherche, et refuse un index périmé.

    Utilisé par ``rag ask`` : on n'interroge jamais un index qui ne correspond
    plus aux passages ou aux options actuels.
    """
    parametres = parametres_encodage(cfg.decoupage, cfg.embeddings, revision)
    empreinte = calculer_empreinte(
        passages, textes_encodes(passages, cfg.decoupage, cfg.embeddings), parametres
    )
    dossier = dossier_index(cfg.stockage.dossier, cfg.decoupage, cfg.embeddings)
    meta = lire_meta(dossier)
    if meta is None or meta.get("empreinte") != empreinte:
        raison = _raison_reconstruction(meta, parametres)
        raise IndexPerime(f"index absent ou périmé ({raison}) : relancer « rag ingest »")
    return IndexVectoriel.charger(dossier), meta


class IndexPerime(RuntimeError):
    pass
