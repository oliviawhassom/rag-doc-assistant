"""EF03 — contrôles avant lecture : format, taille, signature."""

from pathlib import Path

import pytest

from rag.config import IngestionConfig
from rag.ingestion.errors import EmptyFile, FileTooLarge, UnsupportedFormat
from rag.ingestion.validation import validate_file


@pytest.mark.parametrize(("nom", "attendu"), [("a.pdf", "pdf"), ("b.TXT", "txt"), ("c.md", "md")])
def test_formats_acceptes(tmp_path: Path, nom: str, attendu: str) -> None:
    f = tmp_path / nom
    f.write_bytes(b"%PDF-1.4\n" if attendu == "pdf" else b"bonjour")
    assert validate_file(f, IngestionConfig()) == attendu


@pytest.mark.parametrize("nom", ["rapport.docx", "image.png", "sans_extension"])
def test_mauvais_format_rejete(tmp_path: Path, nom: str) -> None:
    f = tmp_path / nom
    f.write_bytes(b"contenu")
    with pytest.raises(UnsupportedFormat, match="non accepté"):
        validate_file(f, IngestionConfig())


def test_extension_desactivee_par_configuration(tmp_path: Path) -> None:
    f = tmp_path / "notes.md"
    f.write_text("# Titre")
    with pytest.raises(UnsupportedFormat):
        validate_file(f, IngestionConfig(extensions_acceptees=(".pdf",)))


def test_faux_pdf_rejete(tmp_path: Path) -> None:
    f = tmp_path / "faux.pdf"
    f.write_text("Je suis un fichier texte renommé en .pdf", encoding="utf-8")
    with pytest.raises(UnsupportedFormat, match="n'est pas un PDF"):
        validate_file(f, IngestionConfig())


def test_fichier_trop_lourd(tmp_path: Path) -> None:
    f = tmp_path / "gros.txt"
    f.write_bytes(b"a" * 2048)
    cfg = IngestionConfig(taille_max_mo=1 / 1024)  # 1 Ko, pour ne pas créer 50 Mo
    with pytest.raises(FileTooLarge, match="trop lourd") as exc:
        validate_file(f, cfg)
    assert exc.value.http_status == 413


def test_limite_par_defaut_50_mo() -> None:
    assert IngestionConfig().taille_max_octets == 50 * 1024 * 1024


def test_fichier_vide(tmp_path: Path) -> None:
    f = tmp_path / "vide.txt"
    f.touch()
    with pytest.raises(EmptyFile):
        validate_file(f, IngestionConfig())
