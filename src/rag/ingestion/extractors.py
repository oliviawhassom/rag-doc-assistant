"""Extraction du texte brut, page par page (EF01, EF02).

Contrat : ``extract_pages`` renvoie une liste de chaînes où l'élément d'indice
``i`` est le texte de la page ``i + 1``. Une page vide reste dans la liste
(chaîne vide) : on ne décale jamais la numérotation, sinon les citations
pointeraient vers la mauvaise page.

Choix de bibliothèque : pypdf (licence BSD, pur Python, pas de dépendance
système). PyMuPDF extrait plus vite et parfois mieux, mais sa licence AGPL est
incompatible avec un dépôt sous MIT. pdfplumber gère mieux la mise en page mais
est nettement plus lent. Si la qualité d'extraction devient un problème, seul
``_extract_pdf`` est à remplacer.

Notion de « page » pour TXT et Markdown : ces formats n'ont pas de pages. Un
fichier texte est une seule page, sauf s'il contient des sauts de page (\\f),
convention historique qui le découpe en plusieurs pages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PasswordType, PdfReader

from rag.ingestion.errors import InvalidEncoding, PasswordProtectedPdf, UnreadablePdf
from rag.ingestion.validation import Format

logger = logging.getLogger(__name__)

# Encodages tentés dans l'ordre pour TXT/MD. utf-8-sig accepte et retire un BOM.
# cp1252 couvre les vieux fichiers Windows ; contrairement à latin-1 il échoue
# sur certains octets, ce qui permet encore de détecter un fichier binaire.
_ENCODAGES = ("utf-8-sig", "cp1252")


AVERTISSEMENT_PDF_CHIFFRE = (
    "PDF chiffré sans mot de passe d'ouverture (restrictions d'usage seulement) : "
    "accepté, texte extrait normalement."
)


@dataclass
class Extraction:
    pages: list[str]
    # Points à signaler sans rejeter le fichier ; le pipeline les journalise.
    avertissements: list[str] = field(default_factory=list)


def extract_document(chemin: Path, fmt: Format) -> Extraction:
    if fmt == "pdf":
        return _extract_pdf(chemin)
    return Extraction(_extract_text(chemin))


def extract_pages(chemin: Path, fmt: Format) -> list[str]:
    """Raccourci quand seules les pages comptent."""
    return extract_document(chemin, fmt).pages


def _extract_pdf(chemin: Path) -> Extraction:
    try:
        reader = PdfReader(chemin)
    except Exception as exc:  # pypdf lève des exceptions variées sur un fichier corrompu
        raise UnreadablePdf(f"PDF illisible ou corrompu : {exc}") from exc

    avertissements: list[str] = []
    if reader.is_encrypted:
        # EF03 : on rejette ce qu'on ne peut pas lire. Un PDF qui n'a qu'un mot de
        # passe propriétaire (restrictions d'impression/copie) s'ouvre avec un mot
        # de passe utilisateur vide : il est accepté, avec un avertissement.
        # Seuls les PDF qui exigent un mot de passe d'ouverture sont rejetés.
        try:
            resultat = reader.decrypt("")
        except Exception as exc:
            raise PasswordProtectedPdf(f"PDF chiffré, impossible à déchiffrer : {exc}") from exc
        if resultat == PasswordType.NOT_DECRYPTED:
            raise PasswordProtectedPdf(
                "PDF protégé par un mot de passe d'ouverture : impossible de lire son contenu."
            )
        avertissements.append(AVERTISSEMENT_PDF_CHIFFRE)

    pages: list[str] = []
    try:
        for page in reader.pages:
            pages.append(page.extract_text() or "")
    except Exception as exc:
        raise UnreadablePdf(f"Erreur à l'extraction de la page {len(pages) + 1} : {exc}") from exc

    if not pages:
        raise UnreadablePdf("Le PDF ne contient aucune page.")
    return Extraction(pages, avertissements)


def _extract_text(chemin: Path) -> list[str]:
    donnees = chemin.read_bytes()
    if b"\x00" in donnees:
        raise InvalidEncoding(
            "Le fichier contient des octets nuls : ce n'est pas un fichier texte."
        )

    for encodage in _ENCODAGES:
        try:
            texte = donnees.decode(encodage)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise InvalidEncoding(f"Encodage non reconnu (essayés : {', '.join(_ENCODAGES)}).")

    texte = texte.replace("\r\n", "\n").replace("\r", "\n")
    return texte.split("\f")
