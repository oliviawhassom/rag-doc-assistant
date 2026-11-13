"""Détection de la langue d'une question : français ou anglais (EF20).

La langue est détectée par le code, pas par le LLM : le message d'abstention
doit être dans la bonne langue même quand la règle du seuil s'applique avant
tout appel au LLM.

Méthode : on compte, dans la question, les mots outils propres à chaque langue
(« comment », « le », « est »… ; « how », « the », « is »…) et les lettres
accentuées, qui n'existent pas en anglais. La langue qui l'emporte nettement
est retenue ; sinon (question très courte, ou faite seulement de termes
techniques comme « kubectl apply ») la langue est incertaine et l'appelant
utilise la langue par défaut de la configuration.

Seules ces deux langues sont gérées, car le corpus est en anglais et les
questions du jeu de test sont en anglais ou en français (D2).
"""

from __future__ import annotations

import re
from typing import Literal

Langue = Literal["fr", "en"]

_MOTS_FR = frozenset(
    """
    le la les un une des du de au aux et ou est sont être ai avons ont pour par
    dans sur avec sans que qui quoi quel quelle quels quelles comment pourquoi
    combien quand où ce cette ces mon ma mes ton ta tes son sa ses notre nos
    votre vos leur leurs je tu il elle nous vous ils elles ne pas plus peut en y
    peux faut doit faire fait entre lorsque si est-ce qu
    """.split()
)
_MOTS_EN = frozenset(
    """
    the an and or is are be been was were for by in with without that which
    what who whom whose how why when where this these those my your his her its
    our their i you he she we they not can could should would must do does did
    to of from into between if there it
    """.split()
)
_ACCENTS = re.compile(r"[àâäçéèêëîïôöùûüÿœæ]", re.IGNORECASE)
_MOT = re.compile(r"[a-zàâäçéèêëîïôöùûüÿœæ]+(?:-[a-zàâäçéèêëîïôöùûüÿœæ]+)*", re.IGNORECASE)

# Un point d'écart suffit : les listes ne contiennent que des mots propres à une
# langue (« a » et « on », qui existent dans les deux, en ont été retirés).
ECART_MINIMAL = 1


def detecter_langue(texte: str) -> Langue | None:
    """« fr », « en », ou None si la langue est incertaine."""
    # « qu'un », « l'index », « d'un » : l'élision est un indice français.
    elisions = len(re.findall(r"\b(?:qu|l|d|j|n|s|c)['’]\w", texte, re.IGNORECASE))
    mots = [m.lower() for m in _MOT.findall(texte.replace("’", "'"))]
    score_fr = sum(m in _MOTS_FR for m in mots) + len(_ACCENTS.findall(texte)) + elisions
    score_en = sum(m in _MOTS_EN for m in mots)
    if score_fr - score_en >= ECART_MINIMAL:
        return "fr"
    if score_en - score_fr >= ECART_MINIMAL:
        return "en"
    return None


def langue_question(texte: str, defaut: Langue) -> tuple[Langue, bool]:
    """Langue retenue et indicateur « détectée » (False = langue par défaut)."""
    detectee = detecter_langue(texte)
    return (detectee, True) if detectee else (defaut, False)
