"""Commandes ``rag ingest`` et ``rag ask``, chargement de la configuration."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import rag.cli
from rag.chunking import TokenizerIndisponible
from rag.cli import main
from rag.config import EmbeddingsConfig, TokenizerConfig, load_config
from rag.index import ModeleIndisponible
from tests.conftest import EncodeurFactice, creer_tokenizer_factice

RACINE = Path(__file__).resolve().parent.parent


REVISION = "e" * 40


@pytest.fixture(autouse=True)
def composants_factices(monkeypatch: pytest.MonkeyPatch) -> list[EncodeurFactice]:
    """La commande charge le vrai tokenizer et le vrai modèle (téléchargements) :
    on les remplace ici. Renvoie la liste des encodeurs effectivement chargés."""
    charges: list[EncodeurFactice] = []

    def charger(cfg: EmbeddingsConfig, revision: str) -> EncodeurFactice:
        encodeur = EncodeurFactice(revision=revision, modele=cfg.modele)
        charges.append(encodeur)
        return encodeur

    monkeypatch.setattr(rag.cli, "charger_tokenizer", lambda cfg: creer_tokenizer_factice())
    monkeypatch.setattr(rag.cli, "resoudre_revision", lambda modele, revision: REVISION)
    monkeypatch.setattr(rag.cli, "charger_encodeur", charger)
    return charges


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
    assert cfg.decoupage.taille == 256
    assert cfg.decoupage.sections_exclues == ("What's next",)
    assert cfg.decoupage.contexte_section is True


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


def test_ingest_decoupe_et_affiche_le_resume(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text(
        "# A\n\nIntro.\n\n## Usage\n\nTexte.\n\n## What's next\n\nLiens.", encoding="utf-8"
    )
    assert main(["ingest", str(corpus), "--config", str(_ecrire_config(tmp_path))]) == 0
    sortie = capsys.readouterr().out
    assert "Découpage taille_fixe-256-32-" in sortie
    assert "2 passages" in sortie
    assert "1 sections exclues (What's next : 1)" in sortie
    assert list((tmp_path / "store" / "passages").iterdir())


def test_tokenizer_indisponible_code_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def indisponible(cfg: TokenizerConfig) -> None:
        raise TokenizerIndisponible("hors ligne")

    monkeypatch.setattr(rag.cli, "charger_tokenizer", indisponible)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("# A", encoding="utf-8")
    assert main(["ingest", str(corpus), "--config", str(_ecrire_config(tmp_path))]) == 2
    assert "Découpage impossible" in capsys.readouterr().err


def test_document_mis_a_jour_affiche_lisiblement(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Le libellé « MIS À JOUR » ne doit pas être collé au chemin du fichier."""
    import json

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("# A\n\nTexte.", encoding="utf-8")
    config = str(_ecrire_config(tmp_path))
    main(["ingest", str(corpus), "--config", config])
    registre = tmp_path / "store" / "documents.jsonl"
    entree = json.loads(registre.read_text(encoding="utf-8"))
    entree["version_nettoyage"] = 1
    registre.write_text(json.dumps(entree) + "\n", encoding="utf-8")
    capsys.readouterr()

    main(["ingest", str(corpus), "--config", config])
    sortie = capsys.readouterr().out
    assert f"MIS À JOUR {corpus / 'a.md'} : Nettoyage v1 → v" in sortie
    assert "Mis à jour : 1" in sortie


# --- Indexation et recherche (module 3) -------------------------------------------------------

CORPUS_MD = {
    "pods.md": "# Pods\n\n## Creating a Pod\n\nCreate a Pod with kubectl apply and a manifest.",
    "services.md": "# Services\n\n## Exposing\n\nA Service exposes an application on the network.",
}


def _corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for nom, texte in CORPUS_MD.items():
        (corpus / nom).write_text(texte, encoding="utf-8")
    return corpus


def test_ingest_construit_puis_recharge_l_index(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], composants_factices: list[EncodeurFactice]
) -> None:
    corpus, config = _corpus(tmp_path), str(_ecrire_config(tmp_path))
    assert main(["ingest", str(corpus), "--config", config]) == 0
    sortie = capsys.readouterr().out
    assert (
        "Index intfloat/multilingual-e5-small : construit (aucun index pour cette configuration)"
        in sortie
    )
    assert "2 passages" in sortie and "0 passage(s) tronqué(s)" in sortie
    assert len(composants_factices) == 1

    assert main(["ingest", str(corpus), "--config", config]) == 0
    assert "rechargé en" in capsys.readouterr().out
    assert len(composants_factices) == 1, "index à jour : le modèle n'est pas rechargé"


def test_ask_affiche_les_passages_et_scores(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus, config = _corpus(tmp_path), str(_ecrire_config(tmp_path))
    main(["ingest", str(corpus), "--config", config])
    capsys.readouterr()

    assert main(["ask", "kubectl apply Pod manifest", "--config", config, "--k", "2"]) == 0
    sortie = capsys.readouterr().out
    lignes = [ligne for ligne in sortie.splitlines() if ligne[:2] in ("1.", "2.")]
    assert len(lignes) == 2
    assert lignes[0].split()[2:] == ["pods.md", "·", "Creating", "a", "Pod"]
    assert "Create a Pod with kubectl apply" in sortie
    assert f"révision {REVISION[:10]}" in sortie


def test_ask_avant_ingest_code_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ask", "Pod ?", "--config", str(_ecrire_config(tmp_path))]) == 2
    assert "rag ingest" in capsys.readouterr().err


def test_ask_refuse_un_index_perime(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Passages modifiés après l'indexation : la recherche refuse l'index périmé."""
    corpus, config = _corpus(tmp_path), str(_ecrire_config(tmp_path))
    main(["ingest", str(corpus), "--config", config])
    # L'ordre des fichiers d'un dossier dépend du système (Windows ≠ Linux) : on
    # modifie le fichier qui contient « kubectl », et on vérifie qu'il y en a un.
    (dossier_passages,) = (tmp_path / "store" / "passages").iterdir()
    modifies = 0
    for fichier in sorted(dossier_passages.glob("*.jsonl")):
        texte = fichier.read_text(encoding="utf-8")
        if "kubectl" in texte:
            fichier.write_text(texte.replace("kubectl", "helm"), encoding="utf-8")
            modifies += 1
    assert modifies == 1, "pré-condition : un fichier de passages a bien été modifié"
    capsys.readouterr()

    assert main(["ask", "Pod ?", "--config", config]) == 2
    assert "périmé" in capsys.readouterr().err


def test_modele_indisponible_code_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def indisponible(modele: str, revision: str) -> str:
        raise ModeleIndisponible("hors ligne")

    monkeypatch.setattr(rag.cli, "resoudre_revision", indisponible)
    assert main(["ingest", str(_corpus(tmp_path)), "--config", str(_ecrire_config(tmp_path))]) == 2
    assert "Indexation impossible" in capsys.readouterr().err
