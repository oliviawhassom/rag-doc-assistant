"""Stockage des passages, un dossier par configuration de découpage.

    store/passages/<config_decoupage>/
      meta.json                  configuration, révision du tokenizer, statistiques
      <document_id>.jsonl        un Passage par ligne, dans l'ordre du document

Un dossier par configuration : les expériences (256, 512, 1 024 tokens ; taille
fixe ou par titres) coexistent sans s'écraser, et chaque passage porte le
même identifiant de configuration que son dossier.

L'écriture est faite dans un dossier temporaire, puis substituée d'un coup :
un arrêt en cours de découpage ne laisse jamais un mélange d'ancien et de
nouveau découpage.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from rag.models import Passage


class PassageStore:
    def __init__(self, dossier_store: Path, config_decoupage: str) -> None:
        self.dossier = dossier_store / "passages" / config_decoupage

    def existe(self) -> bool:
        return (self.dossier / "meta.json").exists()

    def ecrire(self, par_document: dict[str, list[Passage]], meta: dict[str, Any]) -> None:
        temporaire = self.dossier.with_name(self.dossier.name + ".tmp")
        ancien = self.dossier.with_name(self.dossier.name + ".old")
        for d in (temporaire, ancien):
            if d.exists():
                shutil.rmtree(d)
        temporaire.mkdir(parents=True)

        for document_id, passages in par_document.items():
            with (temporaire / f"{document_id}.jsonl").open("w", encoding="utf-8") as f:
                for p in passages:
                    f.write(p.model_dump_json() + "\n")
        (temporaire / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Sous Windows, on ne peut pas renommer par-dessus un dossier existant :
        # on écarte l'ancien, on met le nouveau en place, puis on supprime l'ancien.
        if self.dossier.exists():
            self.dossier.rename(ancien)
        temporaire.rename(self.dossier)
        if ancien.exists():
            shutil.rmtree(ancien)

    def meta(self) -> dict[str, Any]:
        contenu: dict[str, Any] = json.loads(
            (self.dossier / "meta.json").read_text(encoding="utf-8")
        )
        return contenu

    def lire(self, document_id: str) -> list[Passage]:
        with (self.dossier / f"{document_id}.jsonl").open(encoding="utf-8") as f:
            return [Passage.model_validate_json(ligne) for ligne in f if ligne.strip()]

    def lire_tous(self) -> list[Passage]:
        passages: list[Passage] = []
        for fichier in sorted(self.dossier.glob("*.jsonl")):
            passages.extend(self.lire(fichier.stem))
        return passages
