"""Chargement et validation du fichier de configuration (ENF15).

Chaque section du YAML correspond à un modèle Pydantic. Les modèles refusent les
clés inconnues (``extra="forbid"``) : une faute de frappe dans la configuration
provoque une erreur au lieu d'être ignorée, ce qui protège la reproductibilité.
Les modèles sont figés (``frozen=True``) : une configuration ne change pas en
cours d'exécution.

Les sections des modules suivants (découpage, embeddings, recherche...) seront
ajoutées ici au fur et à mesure.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NettoyageConfig(_StrictModel):
    entetes_pieds: bool = True
    lignes_bord: int = Field(default=2, ge=1, le=5)
    seuil_repetition: float = Field(default=0.5, gt=0, le=1)
    min_pages_detection: int = Field(default=3, ge=2)
    cesures: bool = True
    markdown_hugo: bool = True
    variables_hugo: dict[str, str] = Field(default_factory=dict)


class IngestionConfig(_StrictModel):
    extensions_acceptees: tuple[str, ...] = (".pdf", ".txt", ".md")
    taille_max_mo: float = Field(default=50, gt=0)
    min_caracteres_par_page: int = Field(default=20, ge=0)
    nettoyage: NettoyageConfig = Field(default_factory=NettoyageConfig)

    @field_validator("extensions_acceptees")
    @classmethod
    def _normaliser_extensions(cls, valeurs: tuple[str, ...]) -> tuple[str, ...]:
        normalisees = []
        for v in valeurs:
            v = v.strip().lower()
            normalisees.append(v if v.startswith(".") else f".{v}")
        return tuple(normalisees)

    @property
    def taille_max_octets(self) -> int:
        return int(self.taille_max_mo * 1024 * 1024)


class StockageConfig(_StrictModel):
    dossier: Path = Path("store")


class Config(_StrictModel):
    version: str
    stockage: StockageConfig = Field(default_factory=StockageConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)


def load_config(chemin: Path) -> Config:
    """Lit un fichier YAML et renvoie une configuration validée.

    Lève ``FileNotFoundError`` si le fichier n'existe pas et
    ``pydantic.ValidationError`` si son contenu est invalide.
    """
    with chemin.open(encoding="utf-8") as f:
        brut = yaml.safe_load(f) or {}
    if not isinstance(brut, dict):
        raise ValueError(f"{chemin} : la configuration doit être un dictionnaire YAML")
    return Config.model_validate(brut)
