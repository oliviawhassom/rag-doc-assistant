"""Modèles de données (section 4.2 du cahier des charges).

Les noms de champs reprennent exactement ceux du cahier des charges, pour que
les fichiers produits (JSONL) et, plus tard, l'API parlent le même vocabulaire.

Ce module contient pour l'instant ``Document`` et ``Page``. ``Page`` n'est pas
dans le cahier des charges : c'est l'étape intermédiaire entre l'ingestion et le
découpage. Le futur ``Passage`` (avec ``page_debut`` / ``page_fin``) sera
construit à partir des pages.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    """Un fichier importé dans le corpus."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    # Chemin relatif au dossier importé (ex. « concepts/workloads/pods/_index.md »),
    # pas seulement le nom de base : plusieurs fichiers peuvent s'appeler _index.md.
    nom_fichier: str
    source_url: str
    licence: str
    empreinte: str = Field(pattern=r"^[0-9a-f]{64}$", description="SHA-256 du fichier")
    nb_pages: int = Field(ge=1)
    date_ingestion: datetime


class Page(BaseModel):
    """Texte nettoyé d'une page d'un document. Les pages sont numérotées à partir de 1."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    page: int = Field(ge=1)
    texte: str
