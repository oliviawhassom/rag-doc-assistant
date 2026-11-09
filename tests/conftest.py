"""Fixtures partagées.

Les fichiers de test (PDF, TXT, Markdown) sont générés à la volée dans un
dossier temporaire : pas de fichiers binaires dans le dépôt, et chaque test
sait exactement ce que contient chaque page.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from reportlab.lib.pagesizes import A4
from reportlab.lib.pdfencrypt import StandardEncryption
from reportlab.pdfgen import canvas
from tokenizers import Tokenizer, models, pre_tokenizers

from rag.chunking.tokenizer import TokenizerTexte
from rag.config import Config, IngestionConfig, StockageConfig
from rag.ingestion.store import DocumentStore

EN_TETE = "Manuel technique XR-200 — Révision B"
PIED = "Page {n} of {total}"

MakePdf = Callable[..., Path]


def _ecrire_pdf(
    chemin: Path,
    pages: list[list[str]],
    en_tete: str | None = None,
    pied: str | None = None,
    chiffrement: StandardEncryption | None = None,
    sans_texte: bool = False,
) -> Path:
    c = canvas.Canvas(str(chemin), pagesize=A4, encrypt=chiffrement)
    _, hauteur = A4
    total = len(pages)
    for n, lignes in enumerate(pages, start=1):
        if sans_texte:
            # Simule une page numérisée : un dessin, aucun texte.
            c.rect(50, 50, 400, 600, fill=1)
        else:
            c.setFont("Helvetica", 10)
            if en_tete:
                c.drawString(50, hauteur - 40, en_tete)
            y = hauteur - 80
            for ligne in lignes:
                c.drawString(50, y, ligne)
                y -= 14
            if pied:
                c.drawString(250, 30, pied.format(n=n, total=total))
        c.showPage()
    c.save()
    return chemin


@pytest.fixture
def make_pdf(tmp_path: Path) -> MakePdf:
    def _fabrique(nom: str = "doc.pdf", **kwargs: object) -> Path:
        return _ecrire_pdf(tmp_path / nom, **kwargs)  # type: ignore[arg-type]

    return _fabrique


def mot_unique(n: int) -> str:
    """Mot sans chiffre propre à n (4 → « aae »).

    Les lignes de bord ne doivent pas différer d'une page à l'autre que par un
    nombre : le nettoyage les prendrait (à raison) pour un en-tête courant.
    """
    return "".join(chr(97 + int(c)) for c in f"{n:03d}")


def corps_page(n: int) -> list[str]:
    """Contenu reconnaissable pour la page n. Le marqueur est au milieu de la page,
    hors des lignes de bord examinées par la détection d'en-têtes."""
    mot = mot_unique(n)
    return [
        f"Procédure {mot}.",
        f"Introduction du chapitre {mot}.",
        f"MARQUEUR-P{n:03d} : la pression de service du circuit est de {n + 10} bar.",
        "Contrôler les joints avant chaque mise en service.",
        f"Remarque sur l'étape {mot}.",
        f"Fin du chapitre {mot}.",
    ]


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(
        version="test",
        stockage=StockageConfig(dossier=tmp_path / "store"),
        ingestion=IngestionConfig(),
    )


@pytest.fixture
def store(config: Config) -> DocumentStore:
    return DocumentStore(config.stockage.dossier)


def creer_tokenizer_factice() -> TokenizerTexte:
    """Vrai ``tokenizers.Tokenizer`` au vocabulaire vide : chaque mot et chaque signe
    de ponctuation est un token. Même interface que le tokenizer XLM-RoBERTa,
    sans téléchargement ; les comptes sont faciles à vérifier à la main."""
    t = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    return TokenizerTexte(t, "factice", "0" * 40)


@pytest.fixture
def tok() -> TokenizerTexte:
    return creer_tokenizer_factice()


def mots(n: int, prefixe: str = "m") -> str:
    """n mots distincts (donc n tokens pour le tokenizer factice)."""
    return " ".join(f"{prefixe}{i}" for i in range(n))


class EncodeurFactice:
    """Encodeur déterministe, sans modèle : chaque mot est haché vers une dimension.

    Deux textes qui partagent des mots ont des vecteurs proches : les résultats
    de recherche se vérifient à la main. Chaque appel est enregistré, pour
    vérifier exactement ce qui est encodé (préfixes, chemin de section).
    Ne vérifie PAS le calcul d'un vrai modèle : c'est le rôle des tests « slow ».
    """

    def __init__(
        self,
        revision: str = "f" * 40,
        modele: str = "factice/encodeur",
        dimension: int = 256,
        longueur_max: int = 512,
        normaliser: bool = True,
    ) -> None:
        self.modele = modele
        self.revision = revision
        self.dimension = dimension
        self.longueur_max = longueur_max
        self._normaliser = normaliser
        self.appels: list[list[str]] = []

    def encoder(self, textes: list[str], progression: bool = False) -> np.ndarray:
        self.appels.append(list(textes))
        vecteurs = np.zeros((len(textes), self.dimension), dtype=np.float32)
        for i, texte in enumerate(textes):
            for mot in re.findall(r"\w+", texte.lower()):
                vecteurs[
                    i, int(hashlib.sha256(mot.encode()).hexdigest()[:8], 16) % self.dimension
                ] += 1
        if self._normaliser:
            normes = np.linalg.norm(vecteurs, axis=1, keepdims=True)
            vecteurs = vecteurs / np.where(normes == 0, 1, normes)
        return vecteurs.astype(np.float32)

    def compter_tokens(self, texte: str) -> int:
        return len(texte.split()) + 2  # + tokens spéciaux de début et de fin
