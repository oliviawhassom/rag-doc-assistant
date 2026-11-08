"""Contrôles sur le vrai corpus (D3) : aucun chemin de section ambigu.

Ces tests lisent data/corpus/kubernetes, que scripts/download_corpus.py
télécharge. Ils sont sautés si le corpus est absent, par exemple dans
l'intégration continue tant qu'elle ne télécharge pas le corpus : sans lui,
« 0 chemin ambigu » ne prouverait rien.

Si le corpus change (nouveau commit figé), ces tests disent immédiatement si
l'hypothèse « le chemin complet désigne une section unique » tient encore.
"""

from pathlib import Path

import pytest

from rag.chunking.sections import chemins_ambigus, decouper_sections
from rag.config import NettoyageConfig
from rag.ingestion.cleaning import clean_pages
from rag.ingestion.extractors import extract_pages

CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus" / "kubernetes"
FICHIERS = sorted(CORPUS.rglob("*.md")) if CORPUS.exists() else []

pytestmark = pytest.mark.skipif(
    not FICHIERS, reason="corpus absent : lancer scripts/download_corpus.py"
)

# Mêmes réglages que configs/default.yaml.
NETTOYAGE = NettoyageConfig(
    variables_hugo={"currentVersion": "1.35", "latestVersion": "1.35"},
)
EXCLUSIONS = ("What's next",)


def _sections(fichier: Path):  # type: ignore[no-untyped-def]
    pages = clean_pages(extract_pages(fichier, "md"), "md", NETTOYAGE)
    return decouper_sections("\n\n".join(pages), EXCLUSIONS)


def test_aucun_chemin_de_section_ambigu_dans_le_corpus() -> None:
    ambigus = {}
    for fichier in FICHIERS:
        gardees = [s for s in _sections(fichier) if not s.exclue]
        if doublons := chemins_ambigus(gardees):
            ambigus[fichier.relative_to(CORPUS).as_posix()] = doublons
    assert ambigus == {}, f"chemins ambigus : {ambigus}"


def test_chaque_fichier_a_au_moins_une_section() -> None:
    vides = [f.name for f in FICHIERS if not [s for s in _sections(f) if not s.exclue]]
    # Quelques pages d'index ne contiennent qu'un titre : tolérées, mais peu nombreuses.
    assert len(vides) <= 5, vides


def test_aucun_titre_ne_contient_de_residu_de_mise_en_forme() -> None:
    for fichier in FICHIERS:
        for s in _sections(fichier):
            for titre in s.titres:
                assert titre, f"{fichier.name} : titre vide dans « {s.chemin} »"
                for residu in ("`", "**", "{{", "](", "{#", "<"):
                    assert residu not in titre, f"{fichier.name} : « {titre} »"


def test_sections_what_s_next_exclues() -> None:
    exclues = sum(s.exclue for f in FICHIERS for s in _sections(f))
    assert exclues >= 100  # ~120 au commit figé
