"""Écriture atomique d'un dossier de résultats (passages, index).

Le contenu est écrit dans un dossier temporaire, puis mis en place d'un coup :
un arrêt en cours d'écriture ne laisse jamais un mélange d'ancien et de nouveau
contenu. Sous Windows, on ne peut pas renommer par-dessus un dossier existant :
on écarte l'ancien, on met le nouveau en place, puis on supprime l'ancien.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path


def ecrire_dossier_atomique(cible: Path, remplir: Callable[[Path], None]) -> None:
    temporaire = cible.with_name(cible.name + ".tmp")
    ancien = cible.with_name(cible.name + ".old")
    for d in (temporaire, ancien):
        if d.exists():
            shutil.rmtree(d)
    temporaire.mkdir(parents=True)
    remplir(temporaire)
    if cible.exists():
        cible.rename(ancien)
    temporaire.rename(cible)
    if ancien.exists():
        shutil.rmtree(ancien)
