"""Ligne de commande ``rag`` (section 6.1).

``rag ingest <chemin> --config <fichier>``
    Importe les documents, découpe tout le corpus en passages, puis construit
    l'index (ou le recharge s'il est à jour).

``rag ask "<question>" --config <fichier> [--k N]``
    Pour l'instant : affiche les passages retrouvés et leurs scores (EF11). La
    génération de la réponse, avec citations et abstention, arrive au module 4.

``eval`` et ``compare`` viendront avec leurs modules.

Codes de sortie :
    0  succès
    1  ingest : au moins un fichier rejeté (les autres sont importés et indexés)
    2  erreur d'utilisation : configuration invalide, chemin introuvable,
       tokenizer ou modèle impossible à obtenir, index absent ou périmé
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from rag.chunking import (
    PassageStore,
    TokenizerIndisponible,
    charger_tokenizer,
    chemin_affiche,
    decouper_corpus,
)
from rag.config import Config, load_config
from rag.index import (
    IndexPerime,
    ModeleIndisponible,
    MoteurRecherche,
    charger_encodeur,
    charger_index_a_jour,
    indexer,
    resoudre_revision,
)
from rag.ingestion import DocumentStore, ingest_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag", description="Assistant RAG pour documentation technique"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="journalisation détaillée")
    sous = parser.add_subparsers(dest="commande", required=True)

    ingest = sous.add_parser("ingest", help="importe, découpe et indexe les documents")
    ingest.add_argument("chemin", type=Path, help="dossier (ou fichier) à importer")
    ingest.add_argument("--config", type=Path, required=True, help="fichier de configuration YAML")

    ask = sous.add_parser("ask", help="pose une question (pour l'instant : passages retrouvés)")
    ask.add_argument("question", help="la question, entre guillemets")
    ask.add_argument("--config", type=Path, required=True, help="fichier de configuration YAML")
    ask.add_argument(
        "--k", type=int, default=None, help="nombre de passages (défaut : recherche.k)"
    )
    return parser


def _charger_config(chemin: Path) -> Config | None:
    try:
        return load_config(chemin)
    except (OSError, ValueError, ValidationError) as exc:
        print(f"Configuration invalide ({chemin}) :\n{exc}", file=sys.stderr)
        return None


def _cmd_ingest(chemin: Path, chemin_config: Path, detaille: bool) -> int:
    cfg = _charger_config(chemin_config)
    if cfg is None:
        return 2
    documents = DocumentStore(cfg.stockage.dossier)
    try:
        resultats = ingest_path(chemin, cfg, documents)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 2

    for r in resultats:
        if r.statut == "rejete":
            print(f"REJETÉ   {r.chemin} [{r.code_erreur}] {r.message}")
        elif r.statut == "mis_a_jour" or (r.statut == "doublon" and detaille):
            libelle = "MIS À JOUR" if r.statut == "mis_a_jour" else "DOUBLON"
            print(f"{libelle:<11}{r.chemin} : {r.message}")
        for avertissement in r.avertissements:
            print(f"AVERTISSEMENT  {r.chemin} : {avertissement}")

    compte = Counter(r.statut for r in resultats)
    pages = sum(r.document.nb_pages for r in resultats if r.statut == "importe" and r.document)
    print(
        f"\nImportés : {compte['importe']} ({pages} pages) · "
        f"Mis à jour : {compte['mis_a_jour']} · "
        f"Doublons : {compte['doublon']} · Rejetés : {compte['rejete']} · "
        f"Stockage : {cfg.stockage.dossier}"
    )

    # --- Découpage (module 2) ---
    d = cfg.decoupage
    try:
        tok = charger_tokenizer(d.tokenizer)
    except TokenizerIndisponible as exc:
        print(f"Découpage impossible : {exc}", file=sys.stderr)
        return 2
    stats = decouper_corpus(documents, d, tok, tok.revision)
    exclusions = ", ".join(f"{t} : {n}" for t, n in sorted(stats.exclusions_par_titre.items()))
    print(
        f"Découpage {d.identifiant} : {stats.passages} passages · "
        f"{stats.sections} sections, dont {stats.sections_plus_longues} "
        f"({100 * stats.proportion_sections_longues:.1f} %) plus longues que {d.taille} tokens · "
        f"{stats.sections_exclues} sections exclues" + (f" ({exclusions})" if exclusions else "")
    )
    if stats.chemins_ambigus:
        print(
            f"ATTENTION : {len(stats.chemins_ambigus)} chemin(s) de section ambigu(s), "
            "voir le journal"
        )

    # --- Embeddings et index (module 3) ---
    e = cfg.embeddings
    passages = PassageStore(cfg.stockage.dossier, d.identifiant).lire_tous()
    try:
        revision = resoudre_revision(e.modele, e.revision)
        resultat = indexer(
            passages, cfg, revision, lambda: charger_encodeur(e, revision), progression=True
        )
    except ModeleIndisponible as exc:
        print(f"Indexation impossible : {exc}", file=sys.stderr)
        return 2

    m = resultat.meta
    if resultat.statut == "recharge":
        print(
            f"Index {e.modele} : rechargé en {resultat.duree_s:.2f} s "
            f"({m['passages']} passages, empreinte identique)"
        )
    else:
        print(
            f"Index {e.modele} : construit ({resultat.raison}) · {m['passages']} passages "
            f"en {m['duree_encodage_s']:.1f} s ({m['passages_par_seconde']} passages/s) · "
            f"{m['dimension']} dimensions · {m['passages_tronques']} passage(s) tronqué(s)"
        )
    if m["passages_tronques"]:
        print(
            f"ATTENTION : {m['passages_tronques']} passage(s) dépassent la limite de "
            f"{m['longueur_max_modele']} tokens du modèle, voir le journal"
        )
    return 1 if compte["rejete"] else 0


def _cmd_ask(question: str, chemin_config: Path, k: int | None) -> int:
    cfg = _charger_config(chemin_config)
    if cfg is None:
        return 2
    store_passages = PassageStore(cfg.stockage.dossier, cfg.decoupage.identifiant)
    if not store_passages.existe():
        print("Aucun passage pour cette configuration : lancer « rag ingest »", file=sys.stderr)
        return 2
    passages = store_passages.lire_tous()
    e = cfg.embeddings
    try:
        revision = resoudre_revision(e.modele, e.revision)
        debut = time.perf_counter()
        index, _ = charger_index_a_jour(passages, cfg, revision)
        duree_index = time.perf_counter() - debut
        encodeur = charger_encodeur(e, revision)
    except (ModeleIndisponible, IndexPerime) as exc:
        print(f"Recherche impossible : {exc}", file=sys.stderr)
        return 2

    noms = {
        doc.document_id: doc.nom_fichier
        for doc in DocumentStore(cfg.stockage.dossier).list_documents()
    }
    moteur = MoteurRecherche(index, passages, encodeur, e)
    resultats = moteur.rechercher(question, k or cfg.recherche.k)

    print(f"Question : {question}")
    print(f"Index rechargé en {duree_index:.2f} s · {e.modele} (révision {revision[:10]})\n")
    for r in resultats:
        p = r.passage
        lieu = chemin_affiche(p.titre_section) if p.titre_section else f"p. {p.page_debut}"
        extrait = " ".join(p.texte.split())
        extrait = extrait[:220] + ("…" if len(extrait) > 220 else "")
        print(f"{r.rang}. {r.score:.4f}  {noms.get(p.document_id, p.document_id)} · {lieu}")
        print(f"   {extrait}\n")
    print("(Génération de la réponse avec citations : module 4.)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s : %(message)s",
    )
    if args.commande == "ingest":
        return _cmd_ingest(args.chemin, args.config, args.verbose)
    if args.commande == "ask":
        return _cmd_ask(args.question, args.config, args.k)
    return 2  # inaccessible : argparse exige une sous-commande


if __name__ == "__main__":
    sys.exit(main())
