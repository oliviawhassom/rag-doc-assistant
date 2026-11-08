"""Commande ``rag ingest`` et chargement de la configuration."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from rag.cli import main
from rag.config import load_config

RACINE = Path(__file__).resolve().parent.parent


def _ecrire_config(tmp_path: Path, **ingestion: object) -> Path:
    contenu = {"version": "test", "stockage": {"dossier": str(tmp_path / "store")}}
    if ingestion:
        contenu["ingestion"] = ingestion
    chemin = tmp_path / "config.yaml"
    chemin.write_text(yaml.safe_dump(contenu))
    return chemin


def test_config_par_defaut_du_depot_valide() -> None:
    cfg = load_config(RACINE / "configs" / "default.yaml")
    assert cfg.ingestion.taille_max_mo == 50
    assert cfg.ingestion.extensions_acceptees == (".pdf", ".txt", ".md")


def test_cle_inconnue_refusee(tmp_path: Path) -> None:
    chemin = _ecrire_config(tmp_path, taille_max_mb=10)  # faute de frappe : mb au lieu de mo
    with pytest.raises(ValidationError, match="taille_max_mb"):
        load_config(chemin)


def test_ingest_succes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("# A\n\ntexte")
    code = main(["ingest", str(corpus), "--config", str(_ecrire_config(tmp_path))])
    assert code == 0
    assert "Importés : 1 (1 pages)" in capsys.readouterr().out


def test_ingest_avec_rejet(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("# A")
    (corpus / "b.docx").write_text("x")
    code = main(["ingest", str(corpus), "--config", str(_ecrire_config(tmp_path))])
    sortie = capsys.readouterr().out
    assert code == 1
    assert "REJETÉ" in sortie and "b.docx" in sortie and "format_non_accepte" in sortie


def test_ingest_deux_fois_signale_les_doublons(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("# A")
    config = str(_ecrire_config(tmp_path))
    main(["ingest", str(corpus), "--config", config])
    assert main(["ingest", str(corpus), "--config", config]) == 0
    assert "Doublons : 1" in capsys.readouterr().out


def test_config_invalide_code_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    chemin = tmp_path / "cfg.yaml"
    chemin.write_text("version: test\ningestion:\n  taille_max_mo: -3\n")
    assert main(["ingest", str(tmp_path), "--config", str(chemin)]) == 2
    assert "Configuration invalide" in capsys.readouterr().err


def test_chemin_introuvable_code_2(tmp_path: Path) -> None:
    assert (
        main(["ingest", str(tmp_path / "absent"), "--config", str(_ecrire_config(tmp_path))]) == 2
    )


def test_avertissement_affiche(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from reportlab.lib.pdfencrypt import StandardEncryption

    from tests.conftest import _ecrire_pdf, corps_page

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _ecrire_pdf(
        corpus / "restreint.pdf",
        pages=[corps_page(1)],
        chiffrement=StandardEncryption("", ownerPassword="x", canCopy=0),
    )
    assert main(["ingest", str(corpus), "--config", str(_ecrire_config(tmp_path))]) == 0
    sortie = capsys.readouterr().out
    assert "AVERTISSEMENT" in sortie and "restreint.pdf" in sortie
    assert "Importés : 1" in sortie
