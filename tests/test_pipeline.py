"""EF01, EF03, EF04 — enchaînement complet et stockage."""

import json
import logging
from pathlib import Path

import pytest
from reportlab.lib.pdfencrypt import StandardEncryption

from rag.config import Config
from rag.ingestion.extractors import AVERTISSEMENT_PDF_CHIFFRE
from rag.ingestion.pipeline import LICENCE_INCONNUE, ingest_file, ingest_path
from rag.ingestion.store import DocumentStore
from tests.conftest import EN_TETE, PIED, MakePdf, corps_page


def _corpus_trois_formats(dossier: Path, make_pdf: MakePdf) -> None:
    dossier.mkdir()
    pdf = make_pdf(pages=[corps_page(n) for n in range(1, 6)], en_tete=EN_TETE, pied=PIED)
    pdf.rename(dossier / "manuel.pdf")
    (dossier / "notes.txt").write_text("Le couple de serrage est de 25 N.m.", encoding="utf-8")
    (dossier / "guide").mkdir()
    (dossier / "guide" / "_index.md").write_text(
        "# Guide\n\nRedémarrer le service.", encoding="utf-8"
    )


def test_ef01_trois_formats_importes_sans_erreur(
    tmp_path: Path, make_pdf: MakePdf, config: Config, store: DocumentStore
) -> None:
    corpus = tmp_path / "corpus"
    _corpus_trois_formats(corpus, make_pdf)

    resultats = ingest_path(corpus, config, store)

    assert [r.statut for r in resultats] == ["importe"] * 3
    noms = {r.document.nom_fichier: r.document for r in resultats if r.document}
    assert set(noms) == {"guide/_index.md", "manuel.pdf", "notes.txt"}
    assert noms["manuel.pdf"].nb_pages == 5
    pages_pdf = store.load_pages(noms["manuel.pdf"].document_id)
    assert [p.page for p in pages_pdf] == [1, 2, 3, 4, 5]
    assert "MARQUEUR-P004" in pages_pdf[3].texte
    assert "XR-200" not in pages_pdf[3].texte


def test_stockage_persistant_entre_deux_sessions(
    tmp_path: Path, make_pdf: MakePdf, config: Config
) -> None:
    corpus = tmp_path / "corpus"
    _corpus_trois_formats(corpus, make_pdf)
    ingest_path(corpus, config, DocumentStore(config.stockage.dossier))

    relu = DocumentStore(config.stockage.dossier)
    assert len(relu.list_documents()) == 3
    doc = next(d for d in relu.list_documents() if d.nom_fichier == "notes.txt")
    assert relu.load_pages(doc.document_id)[0].texte == "Le couple de serrage est de 25 N.m."


def test_ef04_doublon_non_reindexe(tmp_path: Path, make_pdf: MakePdf, config: Config) -> None:
    corpus = tmp_path / "corpus"
    _corpus_trois_formats(corpus, make_pdf)
    ingest_path(corpus, config, DocumentStore(config.stockage.dossier))
    registre = config.stockage.dossier / "documents.jsonl"
    avant = registre.read_text()

    # Second import du même dossier, dans une nouvelle session.
    resultats = ingest_path(corpus, config, DocumentStore(config.stockage.dossier))

    assert [r.statut for r in resultats] == ["doublon"] * 3
    assert registre.read_text() == avant


def test_ef04_meme_contenu_sous_un_autre_nom(
    tmp_path: Path, config: Config, store: DocumentStore
) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "copie_de_a.txt"
    a.write_text("contenu identique")
    b.write_text("contenu identique")
    premier = ingest_file(a, config, store)
    second = ingest_file(b, config, store)
    assert second.statut == "doublon"
    assert premier.document and premier.document.document_id in second.message


def test_identifiant_deterministe(tmp_path: Path, config: Config) -> None:
    f = tmp_path / "a.txt"
    f.write_text("même contenu", encoding="utf-8")
    id1 = ingest_file(f, config, DocumentStore(tmp_path / "s1")).document
    id2 = ingest_file(f, config, DocumentStore(tmp_path / "s2")).document
    assert id1 and id2 and id1.document_id == id2.document_id


def test_ef03_chaque_rejet_donne_sa_cause(
    tmp_path: Path, make_pdf: MakePdf, config: Config, store: DocumentStore
) -> None:
    dossier = tmp_path / "mauvais"
    dossier.mkdir()
    make_pdf("scan.pdf", pages=[[]] * 3, sans_texte=True).rename(dossier / "scan.pdf")
    make_pdf("chiffre.pdf", pages=[corps_page(1)], chiffrement=StandardEncryption("x")).rename(
        dossier / "chiffre.pdf"
    )
    (dossier / "faux.pdf").write_text("pas un pdf")
    (dossier / "image.png").write_bytes(b"\x89PNG")
    (dossier / "vide.md").write_text("   \n\n")

    resultats = {r.chemin.name: r for r in ingest_path(dossier, config, store)}

    attendu = {
        "scan.pdf": "pdf_sans_texte",
        "chiffre.pdf": "pdf_mot_de_passe_requis",
        "faux.pdf": "format_non_accepte",
        "image.png": "format_non_accepte",
        "vide.md": "fichier_vide",
    }
    for nom, code in attendu.items():
        assert resultats[nom].statut == "rejete", nom
        assert resultats[nom].code_erreur == code, nom
        assert resultats[nom].message, nom
    assert store.list_documents() == []


def test_pdf_dont_seul_l_en_tete_est_du_texte_rejete(
    make_pdf: MakePdf, config: Config, store: DocumentStore
) -> None:
    """Un scan avec un en-tête textuel ajouté par le logiciel : une fois l'en-tête
    retiré, il ne reste rien → rejeté comme PDF sans texte."""
    pdf = make_pdf(pages=[[]] * 4, en_tete=EN_TETE, pied=PIED)
    assert ingest_file(pdf, config, store).code_erreur == "pdf_sans_texte"


def test_page_vide_signalee_mais_conservee(
    make_pdf: MakePdf, config: Config, store: DocumentStore
) -> None:
    pdf = make_pdf(pages=[corps_page(1), [], corps_page(3)])
    r = ingest_file(pdf, config, store)
    assert r.statut == "importe"
    assert r.pages_vides == [2]
    assert r.document and r.document.nb_pages == 3


def test_manifeste_renseigne_source_et_licence(
    tmp_path: Path, config: Config, store: DocumentStore
) -> None:
    corpus = tmp_path / "corpus"
    (corpus / "en").mkdir(parents=True)
    (corpus / "en" / "pods.md").write_text("# Pods")
    (corpus / "sans_entree.md").write_text("# Autre")
    (corpus / "manifest.jsonl").write_text(
        json.dumps(
            {"chemin": "en/pods.md", "source_url": "https://ex.org/pods", "licence": "CC BY 4.0"}
        )
        + "\n"
    )
    (corpus / ".cache").mkdir()
    (corpus / ".cache" / "ignore.md").write_text("caché")

    docs = {
        r.document.nom_fichier: r.document for r in ingest_path(corpus, config, store) if r.document
    }

    assert set(docs) == {"en/pods.md", "sans_entree.md"}  # ni manifeste, ni fichiers cachés
    assert docs["en/pods.md"].source_url == "https://ex.org/pods"
    assert docs["en/pods.md"].licence == "CC BY 4.0"
    assert docs["sans_entree.md"].licence == LICENCE_INCONNUE
    assert docs["sans_entree.md"].source_url.startswith("file://")


# --- EF03 : PDF chiffrés (décision du 8 novembre 2026) ----------------------------


def test_ef03_pdf_mot_de_passe_proprietaire_seul_accepte_avec_avertissement(
    make_pdf: MakePdf,
    config: Config,
    store: DocumentStore,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Mot de passe propriétaire seul (restrictions de copie) : le PDF s'ouvre et son
    texte s'extrait. Il est importé, et un avertissement est journalisé."""
    chiffrement = StandardEncryption("", ownerPassword="proprietaire", canCopy=0, canPrint=0)
    pdf = make_pdf("restreint.pdf", pages=[corps_page(1), corps_page(2)], chiffrement=chiffrement)

    with caplog.at_level(logging.WARNING, logger="rag.ingestion.pipeline"):
        r = ingest_file(pdf, config, store)

    assert r.statut == "importe"
    assert r.avertissements == [AVERTISSEMENT_PDF_CHIFFRE]
    assert r.document is not None
    assert "MARQUEUR-P002" in store.load_pages(r.document.document_id)[1].texte
    journal = [rec for rec in caplog.records if rec.levelno == logging.WARNING]
    assert len(journal) == 1
    assert "restreint.pdf" in journal[0].getMessage()
    assert "sans mot de passe d'ouverture" in journal[0].getMessage()


def test_ef03_pdf_mot_de_passe_ouverture_rejete(
    make_pdf: MakePdf, config: Config, store: DocumentStore
) -> None:
    """Mot de passe d'ouverture : contenu illisible, donc rejeté avec sa cause."""
    chiffrement = StandardEncryption("ouverture", ownerPassword="proprietaire")
    pdf = make_pdf("protege.pdf", pages=[corps_page(1)], chiffrement=chiffrement)

    r = ingest_file(pdf, config, store)

    assert r.statut == "rejete"
    assert r.code_erreur == "pdf_mot_de_passe_requis"
    assert "mot de passe d'ouverture" in r.message
    assert store.list_documents() == []
