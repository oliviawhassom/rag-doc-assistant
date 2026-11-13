"""Lecture de la réponse du LLM : marqueur d'abstention et citations (D14, D15).

Marqueur d'abstention. Le prompt demande au modèle de répondre exactement
``INTROUVABLE_DANS_LES_PASSAGES`` quand les passages ne suffisent pas. Un
modèle ajoute souvent du texte ou de la mise en forme autour : « **Réponse :**
`INTROUVABLE_DANS_LES_PASSAGES`. », minuscules, tirets ou espaces au lieu des
soulignés. La détection normalise donc le texte (majuscules, toute suite de
caractères non alphanumériques remplacée par « _ ») avant de chercher le
marqueur. Une phrase française comme « l'information est introuvable dans les
passages » est aussi reconnue, et c'est voulu : elle exprime la même chose.

Citations. Le modèle cite les passages par numéro : ``[1]``, ``[2][3]``,
``[2, 3]``. Un numéro hors de 1..n (aucun passage ne le porte) est une citation
invalide : retirée de l'affichage, mais comptée et journalisée (D14).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MARQUEUR = "INTROUVABLE_DANS_LES_PASSAGES"

_NON_ALPHANUM = re.compile(r"[^0-9A-Z]+")
_CITATION = re.compile(r"\[\s*(\d+(?:\s*[,;]\s*\d+)*)\s*\]")


def contient_marqueur(texte: str) -> bool:
    normalise = _NON_ALPHANUM.sub("_", texte.upper())
    return MARQUEUR in normalise


@dataclass(frozen=True)
class Citations:
    valides: tuple[int, ...]  # numéros valides, dans l'ordre de première apparition
    invalides: tuple[int, ...]  # numéros cités qui ne correspondent à aucun passage


def extraire_citations(texte: str, nb_passages: int) -> Citations:
    valides: list[int] = []
    invalides: list[int] = []
    for groupe in _CITATION.finditer(texte):
        for brut in re.split(r"[,;]", groupe.group(1)):
            numero = int(brut)
            cible = valides if 1 <= numero <= nb_passages else invalides
            if numero not in cible:
                cible.append(numero)
    return Citations(tuple(valides), tuple(invalides))
