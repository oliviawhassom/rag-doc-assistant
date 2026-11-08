"""Découpage des documents stockés en passages (EF06, EF07).

Pour chaque document du stockage :

    Markdown   → sections (chemin de titres, D3) ; sections exclues comptées et
                 écartées (D7) ; chaque section découpée selon la stratégie (D5)
    PDF, TXT   → pas de titres détectables : fenêtres de taille fixe sur tout le
                 document, en suivant les pages ; si la stratégie « par titres »
                 est demandée, rabattement sur la taille fixe avec avertissement (D6)

Tout le corpus est redécoupé à chaque appel. C'est rapide (quelques secondes) et
cela garantit que les passages reflètent toujours le code et la configuration
actuels, sans risque de passages périmés.

Les statistiques (sections, sections plus longues que la taille, exclusions,
distribution des tailles) sont journalisées et écrites avec les passages : elles
servent à interpréter les expériences (D5 : les stratégies ne diffèrent que sur
les sections longues).
"""

from __future__ import annotations

import bisect
import logging
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any

from rag.chunking.sections import (
    chemins_ambigus,
    decouper_sections,
    normaliser_titre,
)
from rag.chunking.splitter import TOLERANCE, Compteur, fenetres, par_blocs
from rag.chunking.store import PassageStore
from rag.config import DecoupageConfig
from rag.ingestion.store import DocumentStore
from rag.models import Document, Page, Passage

logger = logging.getLogger(__name__)

_EXTENSIONS_MARKDOWN = {".md", ".markdown"}


@dataclass
class StatsDecoupage:
    config_decoupage: str
    taille: int
    documents: int = 0
    sections: int = 0
    sections_plus_longues: int = 0
    sections_exclues: int = 0
    exclusions_par_titre: Counter[str] = field(default_factory=Counter)
    documents_sans_titres: int = 0
    documents_rabattus: list[str] = field(default_factory=list)
    chemins_ambigus: list[str] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)

    @property
    def passages(self) -> int:
        return len(self.tokens)

    @property
    def proportion_sections_longues(self) -> float:
        return self.sections_plus_longues / self.sections if self.sections else 0.0

    def en_dict(self) -> dict[str, Any]:
        borne = (1 + TOLERANCE) * self.taille
        return {
            "config_decoupage": self.config_decoupage,
            "documents": self.documents,
            "sections": self.sections,
            "sections_plus_longues_que_taille": self.sections_plus_longues,
            "proportion_sections_plus_longues": round(self.proportion_sections_longues, 4),
            "sections_exclues": self.sections_exclues,
            "exclusions_par_titre": dict(sorted(self.exclusions_par_titre.items())),
            "documents_sans_titres": self.documents_sans_titres,
            "documents_rabattus_taille_fixe": self.documents_rabattus,
            "chemins_ambigus": self.chemins_ambigus,
            "passages": self.passages,
            "tokens_min": min(self.tokens, default=0),
            "tokens_mediane": statistics.median(self.tokens) if self.tokens else 0,
            "tokens_max": max(self.tokens, default=0),
            "passages_au_dessus_de_la_tolerance": sum(t > borne for t in self.tokens),
        }


def _est_markdown(doc: Document) -> bool:
    return PurePosixPath(doc.nom_fichier).suffix.lower() in _EXTENSIONS_MARKDOWN


def decouper_document(
    doc: Document,
    pages: list[Page],
    cfg: DecoupageConfig,
    tok: Compteur,
    stats: StatsDecoupage,
) -> list[Passage]:
    # Le texte du document est la suite de ses pages ; on garde la position de
    # début de chaque page pour retrouver page_debut et page_fin d'un passage.
    debuts_pages: list[int] = []
    morceaux: list[str] = []
    position = 0
    for page in pages:
        debuts_pages.append(position)
        morceaux.append(page.texte)
        position += len(page.texte) + 2
    texte = "\n\n".join(morceaux)

    def page_de(pos: int) -> int:
        return pages[bisect.bisect_right(debuts_pages, pos) - 1].page

    decoupe = fenetres if cfg.strategie == "taille_fixe" else par_blocs
    intervalles: list[tuple[int, int, str | None]] = []

    if _est_markdown(doc):
        sections = decouper_sections(texte, cfg.sections_exclues)
        exclusions = {normaliser_titre(t) for t in cfg.sections_exclues}
        gardees = []
        for s in sections:
            if s.exclue:
                stats.sections_exclues += 1
                titre = next((t for t in s.titres if normaliser_titre(t) in exclusions), s.chemin)
                stats.exclusions_par_titre[titre] += 1
                continue
            gardees.append(s)
            stats.sections += 1
            if tok.compter(s.texte) > cfg.taille:
                stats.sections_plus_longues += 1
            for d, f in decoupe(s.texte, tok, cfg.taille, cfg.chevauchement):
                intervalles.append((s.debut + d, s.debut + f, s.chemin))
        stats.chemins_ambigus.extend(f"{doc.nom_fichier} : {c}" for c in chemins_ambigus(gardees))
    else:
        stats.documents_sans_titres += 1
        if cfg.strategie == "titres":
            # D6 : pypdf ne donne pas directement les titres d'un PDF.
            logger.warning(
                "%s : pas de titres détectables (PDF ou texte brut) : "
                "découpage par taille fixe au lieu de « par titres »",
                doc.nom_fichier,
            )
            stats.documents_rabattus.append(doc.nom_fichier)
        for d, f in fenetres(texte, tok, cfg.taille, cfg.chevauchement):
            intervalles.append((d, f, None))

    passages: list[Passage] = []
    for debut, fin, chemin in intervalles:
        contenu = texte[debut:fin].strip()
        if not contenu:
            continue
        nb_tokens = tok.compter(contenu)
        stats.tokens.append(nb_tokens)
        passages.append(
            Passage(
                passage_id=f"{doc.document_id}_c{len(passages):04d}",
                document_id=doc.document_id,
                page_debut=page_de(debut),
                page_fin=page_de(fin - 1),
                titre_section=chemin,
                texte=contenu,
                nb_tokens=nb_tokens,
                config_decoupage=cfg.identifiant,
            )
        )
    stats.documents += 1
    return passages


def decouper_corpus(
    documents: DocumentStore, cfg: DecoupageConfig, tok: Compteur, revision_tokenizer: str
) -> StatsDecoupage:
    """Découpe tous les documents stockés et écrit les passages de cette configuration."""
    stats = StatsDecoupage(cfg.identifiant, cfg.taille)
    par_document = {
        doc.document_id: decouper_document(
            doc, documents.load_pages(doc.document_id), cfg, tok, stats
        )
        for doc in documents.list_documents()
    }

    meta = {
        "config_decoupage": cfg.identifiant,
        "decoupage": cfg.model_dump(mode="json"),
        "tokenizer": {"modele": cfg.tokenizer.modele, "revision": revision_tokenizer},
        "date": datetime.now(UTC).isoformat(),
        "statistiques": stats.en_dict(),
    }
    PassageStore(documents.dossier, cfg.identifiant).ecrire(par_document, meta)

    logger.info(
        "Découpage %s : %d passages, %d sections (%.1f %% plus longues que %d tokens), "
        "%d sections exclues %s",
        cfg.identifiant,
        stats.passages,
        stats.sections,
        100 * stats.proportion_sections_longues,
        cfg.taille,
        stats.sections_exclues,
        dict(stats.exclusions_par_titre),
    )
    if stats.chemins_ambigus:
        logger.warning(
            "%d chemin(s) de section ambigu(s) : la citation [fichier, section] ne désigne "
            "pas une section unique : %s",
            len(stats.chemins_ambigus),
            "; ".join(stats.chemins_ambigus),
        )
    return stats


def texte_a_encoder(passage: Passage, cfg: DecoupageConfig) -> str:
    """Texte réellement encodé par le modèle d'embeddings (D5).

    Avec ``contexte_section``, le chemin de section précède le contenu : une
    section courte (« Examples ») gagne le contexte de ses parents. Ce préfixe
    n'est jamais affiché dans les réponses, qui citent ``passage.texte``.
    Le préfixe propre à e5 (« passage: ») est ajouté par le module d'embeddings.
    """
    if cfg.contexte_section and passage.titre_section:
        return f"{passage.titre_section}\n\n{passage.texte}"
    return passage.texte
