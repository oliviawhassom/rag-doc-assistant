"""Tests avec le vrai tokenizer XLM-RoBERTa (D4).

Ils téléchargent le tokenizer.json de bge-m3 au premier lancement (environ
17 Mo, mis en cache ensuite) et sont sautés si Hugging Face est injoignable.
Les autres tests utilisent un tokenizer factice ; ceux-ci vérifient que le
code se comporte de la même façon avec le vrai.
"""

from pathlib import Path

import pytest

from rag.chunking import TokenizerIndisponible, charger_tokenizer
from rag.chunking.decoupage import StatsDecoupage, decouper_document
from rag.chunking.splitter import fenetres
from rag.chunking.tokenizer import TokenizerTexte
from rag.config import Config, DecoupageConfig, TokenizerConfig
from rag.ingestion.pipeline import ingest_file
from rag.ingestion.store import DocumentStore

CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus" / "kubernetes"


@pytest.fixture(scope="module")
def xlmr() -> TokenizerTexte:
    try:
        return charger_tokenizer(TokenizerConfig())
    except TokenizerIndisponible as exc:
        pytest.skip(f"tokenizer réel indisponible : {exc}")


def test_revision_relevee(xlmr: TokenizerTexte) -> None:
    assert len(xlmr.revision) == 40
    assert all(c in "0123456789abcdef" for c in xlmr.revision)


def test_pas_de_troncature(xlmr: TokenizerTexte) -> None:
    """bge-m3 tronque à 8 192 tokens par défaut : le comptage ne doit pas le faire."""
    texte = "Kubernetes schedules Pods onto nodes. " * 3_000
    assert xlmr.compter(texte) > 10_000


def test_positions_couvrent_le_texte(xlmr: TokenizerTexte) -> None:
    texte = "Les Pods « éphémères » ne redémarrent pas : `kubectl get pods`."
    positions = xlmr.positions(texte)
    assert positions[0][0] == 0
    assert positions[-1][1] == len(texte)


def test_ef06_tolerance_avec_le_vrai_tokenizer(xlmr: TokenizerTexte) -> None:
    texte = (
        "A Pod is the smallest deployable unit of computing that you can create and manage "
        "in Kubernetes. Les conteneurs d'un Pod partagent le réseau et le stockage. "
    ) * 200
    for taille, chevauchement in [(256, 32), (512, 64), (1024, 128)]:
        passages = [texte[d:f] for d, f in fenetres(texte, xlmr, taille, chevauchement)]
        comptes = [xlmr.compter(p) for p in passages]
        assert max(comptes) <= 1.1 * taille
        assert min(comptes[:-1]) >= 0.9 * taille


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus absent")
def test_ef06_sur_un_document_du_corpus(
    xlmr: TokenizerTexte, config: Config, store: DocumentStore
) -> None:
    fichier = CORPUS / "en" / "concepts" / "workloads" / "pods" / "pod-lifecycle.md"
    assert ingest_file(fichier, config, store, "pod-lifecycle.md").statut == "importe"
    doc = store.list_documents()[0]
    for strategie in ("taille_fixe", "titres"):
        cfg = DecoupageConfig(strategie=strategie, taille=256, chevauchement=32)
        stats = StatsDecoupage(cfg.identifiant, cfg.taille)
        passages = decouper_document(doc, store.load_pages(doc.document_id), cfg, xlmr, stats)
        assert max(p.nb_tokens for p in passages) <= 1.1 * 256, strategie
        assert stats.chemins_ambigus == []
