"""Enchaînement de l'ingestion : un fichier ou un dossier → documents stockés.

Pour chaque fichier :

    validation (extension, taille, signature)       EF01, EF03
    → empreinte SHA-256 → déjà connue ? doublon     EF04
    → extraction page par page                      EF02, EF03 (chiffré, corrompu)
    → nettoyage                                     EF05
    → contrôle « contient du texte »                EF03 (PDF sans couche texte)
    → enregistrement (pages puis registre)

L'empreinte est calculée avant l'extraction : un doublon est écarté sans payer
le coût de l'extraction.

Une erreur sur un fichier n'arrête pas l'import du dossier : le fichier est
marqué « rejeté » avec sa cause, et on passe au suivant.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from rag.config import Config
from rag.ingestion.cleaning import clean_pages
from rag.ingestion.errors import EmptyFile, IngestionError, PdfWithoutText
from rag.ingestion.extractors import extract_document
from rag.ingestion.store import DocumentStore, document_id_from_hash
from rag.ingestion.validation import Format, validate_file
from rag.models import Document, Page

logger = logging.getLogger(__name__)

MANIFESTE = "manifest.jsonl"
LICENCE_INCONNUE = "non renseignée"

Statut = Literal["importe", "doublon", "rejete"]


@dataclass(frozen=True)
class SourceInfo:
    source_url: str
    licence: str


@dataclass
class IngestionResult:
    chemin: Path
    statut: Statut
    message: str
    document: Document | None = None
    code_erreur: str | None = None
    pages_vides: list[int] = field(default_factory=list)
    avertissements: list[str] = field(default_factory=list)


def sha256_file(chemin: Path, taille_bloc: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with chemin.open("rb") as f:
        while bloc := f.read(taille_bloc):
            h.update(bloc)
    return h.hexdigest()


def load_manifest(dossier: Path) -> dict[str, SourceInfo]:
    """Lit ``manifest.jsonl`` s'il existe à la racine du dossier importé.

    Une ligne par fichier : {"chemin": "<relatif>", "source_url": "...", "licence": "..."}.
    Le script de téléchargement du corpus l'écrit ; c'est ainsi que chaque
    document connaît son URL d'origine et sa licence (section 4.2).
    """
    chemin = dossier / MANIFESTE
    if not chemin.exists():
        return {}
    infos: dict[str, SourceInfo] = {}
    with chemin.open(encoding="utf-8") as f:
        for numero, ligne in enumerate(f, start=1):
            if not ligne.strip():
                continue
            try:
                entree = json.loads(ligne)
                infos[entree["chemin"]] = SourceInfo(entree["source_url"], entree["licence"])
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"{chemin}, ligne {numero} : entrée invalide ({exc})") from exc
    return infos


def _verifier_texte(pages: list[str], fmt: Format, min_par_page: int) -> None:
    nb_caracteres = sum(len("".join(p.split())) for p in pages)
    if fmt == "pdf":
        if nb_caracteres < min_par_page * len(pages):
            raise PdfWithoutText(
                f"Le PDF ne contient presque pas de texte ({nb_caracteres} caractères pour "
                f"{len(pages)} page(s)) : probablement un document numérisé sans couche texte. "
                "L'OCR est hors périmètre."
            )
    elif nb_caracteres == 0:
        raise EmptyFile("Le fichier ne contient aucun texte.")


def ingest_file(
    chemin: Path,
    cfg: Config,
    store: DocumentStore,
    nom_fichier: str | None = None,
    source: SourceInfo | None = None,
) -> IngestionResult:
    nom = nom_fichier or chemin.name
    try:
        fmt = validate_file(chemin, cfg.ingestion)
        empreinte = sha256_file(chemin)

        existant = store.find_by_hash(empreinte)
        if existant is not None:
            return IngestionResult(
                chemin,
                "doublon",
                f"Déjà importé sous {existant.document_id} ({existant.nom_fichier}) : "
                "non réindexé.",
                document=existant,
            )

        extraction = extract_document(chemin, fmt)
        pages = clean_pages(extraction.pages, fmt, cfg.ingestion.nettoyage)
        _verifier_texte(pages, fmt, cfg.ingestion.min_caracteres_par_page)
    except IngestionError as exc:
        logger.warning("%s rejeté [%s] : %s", nom, exc.code, exc.message)
        return IngestionResult(chemin, "rejete", exc.message, code_erreur=exc.code)

    if source is None:
        source = SourceInfo(chemin.resolve().as_uri(), LICENCE_INCONNUE)

    document_id = document_id_from_hash(empreinte)
    doc = Document(
        document_id=document_id,
        nom_fichier=nom,
        source_url=source.source_url,
        licence=source.licence,
        empreinte=empreinte,
        nb_pages=len(pages),
        date_ingestion=datetime.now(UTC),
    )
    store.add(doc, [Page(document_id=document_id, page=i, texte=t) for i, t in enumerate(pages, 1)])

    for avertissement in extraction.avertissements:
        logger.warning("%s : %s", nom, avertissement)
    vides = [i for i, t in enumerate(pages, start=1) if not t.strip()]
    if vides:
        logger.info("%s : pages sans texte %s (conservées pour garder la numérotation)", nom, vides)
    return IngestionResult(
        chemin,
        "importe",
        f"{len(pages)} page(s)",
        document=doc,
        pages_vides=vides,
        avertissements=extraction.avertissements,
    )


def _fichiers(dossier: Path) -> list[Path]:
    """Fichiers du dossier, récursivement, triés (ordre reproductible).
    Ignore les fichiers et dossiers cachés, et le manifeste."""
    fichiers = []
    for chemin in dossier.rglob("*"):
        relatif = chemin.relative_to(dossier)
        if any(partie.startswith(".") for partie in relatif.parts):
            continue
        if chemin.is_file() and relatif.as_posix() != MANIFESTE:
            fichiers.append(chemin)
    return sorted(fichiers, key=lambda p: p.relative_to(dossier).as_posix())


def ingest_path(
    chemin: Path, cfg: Config, store: DocumentStore | None = None
) -> list[IngestionResult]:
    """Importe un fichier seul ou tous les fichiers d'un dossier."""
    store = store or DocumentStore(cfg.stockage.dossier)

    if chemin.is_file():
        return [ingest_file(chemin, cfg, store)]
    if not chemin.is_dir():
        raise FileNotFoundError(f"Chemin introuvable : {chemin}")

    manifeste = load_manifest(chemin)
    if not manifeste:
        logger.warning(
            "Pas de %s dans %s : source_url et licence ne seront pas renseignées",
            MANIFESTE,
            chemin,
        )

    resultats = []
    for fichier in _fichiers(chemin):
        relatif = fichier.relative_to(chemin).as_posix()
        resultats.append(ingest_file(fichier, cfg, store, relatif, manifeste.get(relatif)))
    return resultats
