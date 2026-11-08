"""Contrôles faits AVANT de lire le contenu d'un fichier (EF03).

Ces contrôles sont bon marché (extension, taille, premiers octets) et évitent
de lancer un extracteur sur un fichier qu'on rejettera de toute façon. Les
contrôles qui demandent de lire le document (PDF chiffré, sans texte) sont
faits dans ``extractors`` et ``pipeline``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from rag.config import IngestionConfig
from rag.ingestion.errors import EmptyFile, FileTooLarge, UnsupportedFormat

Format = Literal["pdf", "txt", "md"]

_FORMATS: dict[str, Format] = {".pdf": "pdf", ".txt": "txt", ".md": "md", ".markdown": "md"}

# La norme PDF tolère quelques octets avant l'en-tête « %PDF- ».
_PDF_MAGIC = b"%PDF-"
_PDF_ZONE_EN_TETE = 1024


def validate_file(chemin: Path, cfg: IngestionConfig) -> Format:
    """Vérifie extension, taille et signature ; renvoie le format détecté."""
    extension = chemin.suffix.lower()
    if extension not in cfg.extensions_acceptees or extension not in _FORMATS:
        acceptes = ", ".join(cfg.extensions_acceptees)
        raise UnsupportedFormat(
            f"Format « {extension or 'sans extension'} » non accepté "
            f"(formats acceptés : {acceptes})."
        )
    fmt = _FORMATS[extension]

    taille = chemin.stat().st_size
    if taille == 0:
        raise EmptyFile("Le fichier est vide (0 octet).")
    if taille > cfg.taille_max_octets:
        raise FileTooLarge(
            f"Fichier trop lourd : {taille / 1024 / 1024:.1f} Mo, "
            f"maximum autorisé {cfg.taille_max_mo:g} Mo."
        )

    if fmt == "pdf":
        with chemin.open("rb") as f:
            debut = f.read(_PDF_ZONE_EN_TETE)
        if _PDF_MAGIC not in debut:
            raise UnsupportedFormat(
                "Le fichier porte l'extension .pdf mais son contenu n'est pas un PDF."
            )
    return fmt
