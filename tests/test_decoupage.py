"""EF02, EF05, EF06, EF07, D5–D7 — découpage de documents stockés en passages."""

import json
import logging
import random
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag.chunking import PassageStore, decouper_corpus, texte_a_encoder
from rag.chunking.decoupage import StatsDecoupage, decouper_document
from rag.chunking.tokenizer import TokenizerTexte
from rag.config import Config, DecoupageConfig
from rag.ingestion.pipeline import ingest_file
from rag.ingestion.store import DocumentStore
from rag.models import Passage
from tests.conftest import EN_TETE, PIED, MakePdf, corps_page, mots


def _decouper(
    store: DocumentStore, nom: str, cfg: DecoupageConfig, tok: TokenizerTexte
) -> tuple[list[Passage], StatsDecoupage]:
    doc = next(d for d in store.list_documents() if d.nom_fichier == nom)
    stats = StatsDecoupage(cfg.identifiant, cfg.taille)
    return decouper_document(doc, store.load_pages(doc.document_id), cfg, tok, stats), stats


def _importer_md(
    tmp_path: Path, config: Config, store: DocumentStore, nom: str, texte: str
) -> None:
    chemin = tmp_path / nom
    chemin.write_text(texte, encoding="utf-8")
    assert ingest_file(chemin, config, store).statut == "importe"


GUIDE = f"""---
title: Guide
---

Introduction du guide.

## Installation

{mots(40, "inst")}

### Prérequis `kubectl`

{mots(20, "pre")}

## Utilisation

{mots(300, "util")}

## What's next

* Lien vers une autre page
"""


# --- Markdown : sections, pages, champs du Passage ------------------------------------


def test_passages_markdown(
    tmp_path: Path, config: Config, store: DocumentStore, tok: TokenizerTexte
) -> None:
    _importer_md(tmp_path, config, store, "guide.md", GUIDE)
    cfg = DecoupageConfig(taille=128, chevauchement=16)
    passages, stats = _decouper(store, "guide.md", cfg, tok)

    chemins = [p.titre_section for p in passages]
    assert chemins[:3] == ["(introduction)", "Installation", "Installation > Prérequis kubectl"]
    assert set(chemins[3:]) == {"Utilisation"}
    assert len(passages) == 3 + 3  # 300 mots en fenêtres de 128 / pas de 112
    assert all(p.page_debut == p.page_fin == 1 for p in passages)
    assert all(p.config_decoupage == cfg.identifiant for p in passages)
    assert [p.passage_id for p in passages] == [
        f"{passages[0].document_id}_c{i:04d}" for i in range(len(passages))
    ]
    assert all(p.nb_tokens == tok.compter(p.texte) for p in passages)
    assert not any("Lien vers une autre page" in p.texte for p in passages)
    assert stats.sections == 4
    assert stats.sections_plus_longues == 1
    assert stats.sections_exclues == 1


def test_ef07_strategie_choisie_par_la_configuration(
    tmp_path: Path, config: Config, store: DocumentStore, tok: TokenizerTexte
) -> None:
    paragraphes = "\n\n".join(mots(50, f"p{i}_") for i in range(6))
    _importer_md(tmp_path, config, store, "long.md", f"## Longue\n\n{paragraphes}")
    fixe = DecoupageConfig(strategie="taille_fixe", taille=128, chevauchement=16)
    titres = DecoupageConfig(strategie="titres", taille=128, chevauchement=16)

    p_fixe, _ = _decouper(store, "long.md", fixe, tok)
    p_titres, _ = _decouper(store, "long.md", titres, tok)

    assert fixe.identifiant != titres.identifiant
    assert [p.nb_tokens for p in p_titres] == [100, 100, 100]  # 2 paragraphes par passage
    assert [p.nb_tokens for p in p_fixe] == [128, 128, 76]
    assert all(p.texte.split("\n\n")[0].startswith("p") for p in p_titres)


# --- PDF : pages (EF02) et en-têtes (EF05) rejoués sur les passages --------------------


def _pdf_importe(make_pdf: MakePdf, config: Config, store: DocumentStore, nb_pages: int) -> None:
    pages = [[*corps_page(n), mots(60, f"q{n}_")] for n in range(1, nb_pages + 1)]
    pdf = make_pdf("manuel.pdf", pages=pages, en_tete=EN_TETE, pied=PIED)
    assert ingest_file(pdf, config, store).statut == "importe"


def test_ef02_pages_des_passages_pdf(
    make_pdf: MakePdf, config: Config, store: DocumentStore, tok: TokenizerTexte
) -> None:
    """EF02 sur les passages : pour 20 passages tirés au hasard, chaque marqueur
    MARQUEUR-Pnnn qu'ils contiennent est dans l'intervalle [page_debut, page_fin]."""
    _pdf_importe(make_pdf, config, store, 30)
    passages, _ = _decouper(store, "manuel.pdf", DecoupageConfig(taille=64, chevauchement=8), tok)
    assert len(passages) >= 20
    assert any(p.page_fin > p.page_debut for p in passages), "des passages couvrent 2 pages"

    for p in random.Random(42).sample(passages, 20):
        for n in map(int, re.findall(r"MARQUEUR-P(\d{3})", p.texte)):
            assert p.page_debut <= n <= p.page_fin, (p.passage_id, n)
        assert p.titre_section is None


def test_ef05_aucun_en_tete_dans_20_passages(
    make_pdf: MakePdf, config: Config, store: DocumentStore, tok: TokenizerTexte
) -> None:
    _pdf_importe(make_pdf, config, store, 30)
    passages, _ = _decouper(store, "manuel.pdf", DecoupageConfig(taille=64, chevauchement=8), tok)
    for p in random.Random(7).sample(passages, 20):
        assert "XR-200" not in p.texte
        assert not re.search(r"Page \d+ of \d+", p.texte)


def test_d6_pdf_par_titres_rabattu_sur_taille_fixe(
    make_pdf: MakePdf,
    config: Config,
    store: DocumentStore,
    tok: TokenizerTexte,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _pdf_importe(make_pdf, config, store, 5)
    titres = DecoupageConfig(strategie="titres", taille=64, chevauchement=8)
    fixe = DecoupageConfig(strategie="taille_fixe", taille=64, chevauchement=8)

    with caplog.at_level(logging.WARNING, logger="rag.chunking.decoupage"):
        p_titres, stats = _decouper(store, "manuel.pdf", titres, tok)
    p_fixe, _ = _decouper(store, "manuel.pdf", fixe, tok)

    assert [p.texte for p in p_titres] == [p.texte for p in p_fixe]
    assert stats.documents_rabattus == ["manuel.pdf"]
    assert any("taille fixe" in r.getMessage() for r in caplog.records)


# --- Corpus entier, stockage, statistiques ----------------------------------------------


def test_decouper_corpus_ecrit_passages_et_statistiques(
    tmp_path: Path,
    config: Config,
    store: DocumentStore,
    tok: TokenizerTexte,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _importer_md(tmp_path, config, store, "guide.md", GUIDE)
    _importer_md(tmp_path, config, store, "autre.md", "## A\n\ntexte\n\n## What's next\n\nliens")
    cfg = DecoupageConfig(taille=128, chevauchement=16)

    with caplog.at_level(logging.INFO, logger="rag.chunking.decoupage"):
        stats = decouper_corpus(store, cfg, tok, "abc123")

    # D7 : nombre de sections exclues journalisé à chaque indexation
    journal = " ".join(r.getMessage() for r in caplog.records)
    assert "2 sections exclues" in journal
    assert stats.exclusions_par_titre == {"What's next": 2}

    passages = PassageStore(store.dossier, cfg.identifiant)
    meta = passages.meta()
    assert meta["tokenizer"] == {"modele": "BAAI/bge-m3", "revision": "abc123"}
    assert meta["statistiques"]["sections_exclues"] == 2
    assert meta["statistiques"]["proportion_sections_plus_longues"] == pytest.approx(1 / 5)
    assert meta["statistiques"]["passages_au_dessus_de_la_tolerance"] == 0
    assert len(passages.lire_tous()) == stats.passages


def test_redecoupage_identique_et_sans_residu(
    tmp_path: Path, config: Config, store: DocumentStore, tok: TokenizerTexte
) -> None:
    """ENF06 : deux découpages de la même configuration donnent les mêmes passages,
    et un redécoupage ne laisse ni dossier temporaire ni ancien fichier."""
    _importer_md(tmp_path, config, store, "guide.md", GUIDE)
    cfg = DecoupageConfig(taille=128, chevauchement=16)
    decouper_corpus(store, cfg, tok, "r")
    premier = PassageStore(store.dossier, cfg.identifiant).lire_tous()
    decouper_corpus(store, cfg, tok, "r")
    assert PassageStore(store.dossier, cfg.identifiant).lire_tous() == premier
    assert sorted(p.name for p in (store.dossier / "passages").iterdir()) == [cfg.identifiant]


def test_chemins_ambigus_signales(
    tmp_path: Path,
    config: Config,
    store: DocumentStore,
    tok: TokenizerTexte,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _importer_md(tmp_path, config, store, "dup.md", "## Exemples\n\na\n\n## Exemples\n\nb")
    with caplog.at_level(logging.WARNING):
        stats = decouper_corpus(store, DecoupageConfig(), tok, "r")
    assert stats.chemins_ambigus == ["dup.md : Exemples"]
    assert "ambigu" in caplog.text


# --- Texte encodé (D5) et configuration -------------------------------------------------


def _passage(titre: str | None) -> Passage:
    return Passage(
        passage_id="d_c0000",
        document_id="d",
        page_debut=1,
        page_fin=1,
        titre_section=titre,
        texte="Contenu.",
        nb_tokens=2,
        config_decoupage="x",
    )


def test_contexte_section_en_tete_du_texte_encode() -> None:
    p = _passage("Imperative commands > Examples")
    assert texte_a_encoder(p, DecoupageConfig()) == "Imperative commands > Examples\n\nContenu."
    assert texte_a_encoder(p, DecoupageConfig(contexte_section=False)) == "Contenu."
    assert texte_a_encoder(_passage(None), DecoupageConfig()) == "Contenu."
    assert p.texte == "Contenu.", "le texte affiché et cité ne change pas"


def test_identifiant_de_configuration() -> None:
    base = DecoupageConfig()
    assert base.identifiant == DecoupageConfig().identifiant
    assert base.identifiant.startswith("taille_fixe-256-32-")
    variantes = [
        DecoupageConfig(taille=512),
        DecoupageConfig(strategie="titres"),
        DecoupageConfig(contexte_section=False),
        DecoupageConfig(sections_exclues=()),
        DecoupageConfig(tokenizer={"modele": "BAAI/bge-m3", "revision": "a" * 40}),
    ]
    assert len({base.identifiant, *(v.identifiant for v in variantes)}) == 6


def test_chevauchement_trop_grand_refuse() -> None:
    with pytest.raises(ValidationError, match="chevauchement"):
        DecoupageConfig(taille=100, chevauchement=60)


def test_meta_lisible(
    tmp_path: Path, config: Config, store: DocumentStore, tok: TokenizerTexte
) -> None:
    _importer_md(tmp_path, config, store, "guide.md", GUIDE)
    cfg = DecoupageConfig()
    decouper_corpus(store, cfg, tok, "r")
    brut = (store.dossier / "passages" / cfg.identifiant / "meta.json").read_text(encoding="utf-8")
    assert json.loads(brut)["decoupage"]["sections_exclues"] == ["What's next"]
