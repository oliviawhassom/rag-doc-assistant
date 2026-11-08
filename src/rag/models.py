"""Modèles de données (section 4.2 du cahier des charges).

Les noms de champs reprennent exactement ceux du cahier des charges, pour que
les fichiers produits (JSONL) et, plus tard, l'API parlent le même vocabulaire.

``Page`` n'est pas dans le cahier des charges : c'est l'étape intermédiaire
entre l'ingestion (qui produit des pages) et le découpage (qui produit des
``Passage`` à partir des pages).
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
    # Hors cahier des charges : version du nettoyage qui a produit les pages
    # (rag.ingestion.cleaning.VERSION_NETTOYAGE). Absent d'un ancien registre → 1.
    version_nettoyage: int = 1


class Page(BaseModel):
    """Texte nettoyé d'une page d'un document. Les pages sont numérotées à partir de 1."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    page: int = Field(ge=1)
    texte: str


class Passage(BaseModel):
    """Extrait de document découpé pour la recherche (section 4.2).

    ``titre_section`` est le chemin complet des titres, séparés par « > »
    (décision D3) : « Imperative commands > Examples ». Il vaut ``None`` pour un
    PDF ou un texte brut, où aucun titre n'est détecté ; la citation se fait alors
    par la page. Pour un Markdown, ``page_debut`` = ``page_fin`` = 1.

    ``texte`` est le contenu affiché et cité. Le texte effectivement encodé peut
    être précédé du chemin de section (option ``contexte_section``) : voir
    ``rag.chunking.texte_a_encoder``.
    """

    model_config = ConfigDict(frozen=True)

    passage_id: str
    document_id: str
    page_debut: int = Field(ge=1)
    page_fin: int = Field(ge=1)
    titre_section: str | None
    texte: str = Field(min_length=1)
    nb_tokens: int = Field(ge=1)
    config_decoupage: str
