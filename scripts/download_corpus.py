"""Télécharge le corpus Kubernetes depuis sa source officielle (section 4.1).

Le corpus n'est pas versionné dans le dépôt : ce script le reconstruit à
l'identique, car il récupère toujours le même commit du dépôt kubernetes/website.

    python scripts/download_corpus.py

Le corpus est en anglais uniquement (décision D2, docs/decisions.md) : les
questions en français interrogent le texte anglais.

Résultat :
    data/raw/kubernetes-website/     copie partielle du dépôt (sparse checkout)
    data/corpus/kubernetes/en/<section>/...md
    data/corpus/kubernetes/manifest.jsonl   source_url + licence de chaque fichier

Licence : la documentation Kubernetes est publiée sous CC BY 4.0 (fichier
LICENSE du dépôt). Voir data/README.md.

Seul git est nécessaire (pas de dépendance Python).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

DEPOT = "https://github.com/kubernetes/website.git"
# Commit figé (tête de la branche release-1.35 au 8 novembre 2026) : le corpus est
# identique d'une machine à l'autre. Changer ce commit = changer de version de corpus.
COMMIT = "8817d358864b70a9710c5ad34f78df6a239be33d"
VERSION_DOCS = "1.35"
LICENCE = "CC BY 4.0"
# Le préfixe de langue est gardé dans les chemins (« en/concepts/... ») : les
# noms de fichiers restent stables si une autre langue est ajoutée un jour.
LANGUE = "en"

RACINE = Path(__file__).resolve().parent.parent


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)


def recuperer_depot(destination: Path, chemins: list[str]) -> None:
    if not (destination / ".git").exists():
        destination.mkdir(parents=True, exist_ok=True)
        _git("init", "-q", cwd=destination)
        _git("remote", "add", "origin", DEPOT, cwd=destination)
    _git("sparse-checkout", "set", "--no-cone", *chemins, cwd=destination)
    _git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", COMMIT, cwd=destination)
    _git("checkout", "-q", "FETCH_HEAD", cwd=destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sections", nargs="+", default=["concepts"])
    parser.add_argument("--sortie", type=Path, default=RACINE / "data" / "corpus" / "kubernetes")
    args = parser.parse_args()

    brut = RACINE / "data" / "raw" / "kubernetes-website"
    chemins = [f"content/{LANGUE}/docs/{s}/" for s in args.sections]
    print(f"Téléchargement de kubernetes/website@{COMMIT[:10]} ({', '.join(chemins)})")
    recuperer_depot(brut, chemins)

    if args.sortie.exists():
        shutil.rmtree(args.sortie)
    racine_docs = brut / "content" / LANGUE / "docs"
    entrees = []
    for section in args.sections:
        for fichier in sorted((racine_docs / section).rglob("*.md")):
            relatif_docs = fichier.relative_to(racine_docs).as_posix()
            cible_rel = f"{LANGUE}/{relatif_docs}"
            cible = args.sortie / cible_rel
            cible.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fichier, cible)
            entrees.append(
                {
                    "chemin": cible_rel,
                    "source_url": f"https://github.com/kubernetes/website/blob/{COMMIT}/"
                    f"content/{LANGUE}/docs/{relatif_docs}",
                    "licence": LICENCE,
                }
            )

    with (args.sortie / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for e in entrees:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"{len(entrees)} fichiers copiés dans {args.sortie} (docs Kubernetes {VERSION_DOCS})")


if __name__ == "__main__":
    main()
