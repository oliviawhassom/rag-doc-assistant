"""Chargement et validation du fichier de configuration (ENF15).

Chaque section du YAML correspond à un modèle Pydantic. Les modèles refusent les
clés inconnues (``extra="forbid"``) : une faute de frappe dans la configuration
provoque une erreur au lieu d'être ignorée, ce qui protège la reproductibilité.
Les modèles sont figés (``frozen=True``) : une configuration ne change pas en
cours d'exécution.

Les sections des modules suivants (embeddings, recherche...) seront ajoutées
ici au fur et à mesure.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class TokenizerConfig(_StrictModel):
    """Tokenizer qui définit ce qu'est « un token » pour les tailles de passages (D4).

    ``revision`` : hash du commit Hugging Face. « main » est accepté pour un premier
    lancement, avec un avertissement : la révision réellement téléchargée est alors
    relevée dans les métadonnées du découpage, pour être recopiée ici.
    """

    modele: str = "BAAI/bge-m3"
    revision: str = "main"


class DecoupageConfig(_StrictModel):
    strategie: Literal["taille_fixe", "titres"] = "taille_fixe"
    taille: int = Field(default=256, ge=16, le=8192, description="en tokens")
    chevauchement: int = Field(default=32, ge=0, description="en tokens (taille fixe)")
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    # Sections exclues du découpage, avec leurs sous-sections. Comparaison
    # insensible à la casse et aux variantes d'apostrophe.
    sections_exclues: tuple[str, ...] = ("What's next",)
    # Préfixer le texte encodé de chaque passage par son chemin de section (D2).
    contexte_section: bool = True

    @model_validator(mode="after")
    def _chevauchement_coherent(self) -> DecoupageConfig:
        if self.chevauchement * 2 > self.taille:
            raise ValueError(
                f"chevauchement ({self.chevauchement}) supérieur à la moitié de la taille "
                f"({self.taille}) : chaque token serait encodé plus de deux fois"
            )
        return self

    @property
    def identifiant(self) -> str:
        """Valeur du champ ``config_decoupage`` des passages.

        Lisible (stratégie, taille, chevauchement) et suffixée d'une empreinte de
        toute la section : deux réglages différents ne partagent jamais de passages.
        """
        empreinte = hashlib.sha256(
            json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        ).hexdigest()[:8]
        return f"{self.strategie}-{self.taille}-{self.chevauchement}-{empreinte}"


class StockageConfig(_StrictModel):
    dossier: Path = Path("store")


class Config(_StrictModel):
    version: str
    stockage: StockageConfig = Field(default_factory=StockageConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    decoupage: DecoupageConfig = Field(default_factory=DecoupageConfig)


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
