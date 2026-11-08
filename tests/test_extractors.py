"""EF01, EF02, EF03 — extraction page par page et rejets à la lecture."""

import random
import re
from pathlib import Path

import pytest
from reportlab.lib.pdfencrypt import StandardEncryption

from rag.ingestion.errors import InvalidEncoding, PasswordProtectedPdf, UnreadablePdf
from rag.ingestion.extractors import extract_pages
from tests.conftest import MakePdf, corps_page


def test_ef02_page_correcte_sur_20_pages_tirees_au_hasard(make_pdf: MakePdf) -> None:
    """Critère EF02 : pour 20 pages tirées au hasard, le numéro enregistré est le bon.

    Chaque page porte un marqueur unique « MARQUEUR-Pnnn » ; on vérifie que le
    texte rangé sous le numéro n contient le marqueur de la page n, et lui seul.
    """
    nb_pages = 40
    pdf = make_pdf(pages=[corps_page(n) for n in range(1, nb_pages + 1)])
    pages = extract_pages(pdf, "pdf")
    assert len(pages) == nb_pages

    for n in random.Random(42).sample(range(1, nb_pages + 1), 20):
        marqueurs = re.findall(r"MARQUEUR-P(\d{3})", pages[n - 1])
        assert marqueurs == [f"{n:03d}"], f"page {n} : marqueurs trouvés {marqueurs}"


def test_page_blanche_conservee(make_pdf: MakePdf) -> None:
    """Une page vide garde sa place : la page 3 reste la page 3."""
    pdf = make_pdf(pages=[corps_page(1), [], corps_page(3)])
    pages = extract_pages(pdf, "pdf")
    assert len(pages) == 3
    assert pages[1].strip() == ""
    assert "MARQUEUR-P003" in pages[2]


def test_pdf_protege_par_mot_de_passe_rejete(make_pdf: MakePdf) -> None:
    pdf = make_pdf(pages=[corps_page(1)], chiffrement=StandardEncryption("secret"))
    with pytest.raises(PasswordProtectedPdf, match="mot de passe d'ouverture"):
        extract_pages(pdf, "pdf")


def test_pdf_chiffre_sans_mot_de_passe_utilisateur_accepte(make_pdf: MakePdf) -> None:
    """Chiffré seulement pour restreindre impression/copie : lisible, donc accepté."""
    chiffrement = StandardEncryption("", ownerPassword="proprietaire", canCopy=0)
    pdf = make_pdf(pages=[corps_page(1)], chiffrement=chiffrement)
    assert "MARQUEUR-P001" in extract_pages(pdf, "pdf")[0]


def test_pdf_corrompu_rejete(make_pdf: MakePdf, tmp_path: Path) -> None:
    pdf = make_pdf(pages=[corps_page(1)])
    tronque = tmp_path / "tronque.pdf"
    tronque.write_bytes(pdf.read_bytes()[:200])
    with pytest.raises(UnreadablePdf):
        extract_pages(tronque, "pdf")


def test_txt_une_page_par_defaut(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    # write_bytes : en mode texte, Windows convertirait chaque « \n » en « \r\n ».
    f.write_bytes(b"ligne 1\r\nligne 2\n")
    assert extract_pages(f, "txt") == ["ligne 1\nligne 2\n"]


def test_txt_saut_de_page_cree_une_page(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("page un\fpage deux\fpage trois", encoding="utf-8")
    assert extract_pages(f, "txt") == ["page un", "page deux", "page trois"]


def test_markdown_avec_bom_utf8(tmp_path: Path) -> None:
    f = tmp_path / "a.md"
    f.write_bytes("﻿# Présentation".encode())
    assert extract_pages(f, "md") == ["# Présentation"]


def test_txt_windows_1252(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_bytes("Pression « maximale » : 12 bar, coût 5 €".encode("cp1252"))
    assert extract_pages(f, "txt") == ["Pression « maximale » : 12 bar, coût 5 €"]


def test_binaire_deguise_en_txt_rejete(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00")
    with pytest.raises(InvalidEncoding):
        extract_pages(f, "txt")
