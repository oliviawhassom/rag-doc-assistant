"""Ligne de commande ``rag`` (section 6.1).

Pour l'instant une seule sous-commande : ``rag ingest <chemin> --config <fichier>``.
``ask``, ``eval`` et ``compare`` viendront avec leurs modules.

Codes de sortie :
    0  tout est importé (ou déjà présent)
    1  au moins un fichier rejeté (les autres sont quand même importés)
    2  erreur d'utilisation : configuration invalide, chemin introuvable
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from rag.config import load_config
from rag.ingestion import ingest_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag", description="Assistant RAG pour documentation technique"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="journalisation détaillée")
    sous = parser.add_subparsers(dest="commande", required=True)

    ingest = sous.add_parser("ingest", help="importe et stocke les documents d'un dossier")
    ingest.add_argument("chemin", type=Path, help="dossier (ou fichier) à importer")
    ingest.add_argument("--config", type=Path, required=True, help="fichier de configuration YAML")
    return parser


def _cmd_ingest(chemin: Path, chemin_config: Path) -> int:
    try:
        cfg = load_config(chemin_config)
    except (OSError, ValueError, ValidationError) as exc:
        print(f"Configuration invalide ({chemin_config}) :\n{exc}", file=sys.stderr)
        return 2
    try:
        resultats = ingest_path(chemin, cfg)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 2

    for r in resultats:
        if r.statut == "rejete":
            print(f"REJETÉ   {r.chemin} [{r.code_erreur}] {r.message}")
        elif r.statut == "doublon":
            print(f"DOUBLON  {r.chemin} : {r.message}")
        for avertissement in r.avertissements:
            print(f"AVERTISSEMENT  {r.chemin} : {avertissement}")

    compte = Counter(r.statut for r in resultats)
    pages = sum(r.document.nb_pages for r in resultats if r.statut == "importe" and r.document)
    print(
        f"\nImportés : {compte['importe']} ({pages} pages) · "
        f"Doublons : {compte['doublon']} · Rejetés : {compte['rejete']} · "
        f"Stockage : {cfg.stockage.dossier}"
    )
    return 1 if compte["rejete"] else 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s : %(message)s",
    )
    if args.commande == "ingest":
        return _cmd_ingest(args.chemin, args.config)
    return 2  # inaccessible : argparse exige une sous-commande


if __name__ == "__main__":
    sys.exit(main())
