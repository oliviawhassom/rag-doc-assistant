"""Construction du prompt envoyé au LLM (EF15, EF21, décisions D16, D17).

Prompts versionnés (D16) : ``prompts/<version>/systeme.txt`` et
``prompts/<version>/gabarit.txt``. La version et l'empreinte SHA-256 de chacun
sont jointes au prompt construit, puis à chaque réponse journalisée.

Passages numérotés : le LLM cite ``[1]``, ``[2]``… ; le code convertit ensuite
ces numéros en [fichier, section] ou [document, page] (D14).

Neutralisation (D17) : les passages sont placés entre balises
``<passage numero="n"> … </passage>``, dans une zone ``<passages>``. Un
document pourrait contenir lui-même ``</passage>`` ou ``</passages>`` pour
« sortir » de la zone de données et faire passer la suite pour des
instructions. Avant de construire le prompt, toute balise de ce type présente
dans un passage, ou dans la question, est rendue inoffensive : ses chevrons
``<`` ``>`` deviennent ``‹`` ``›``. Seules ces balises sont touchées : un
``<pod-name>`` dans une commande reste intact. Le passage stocké et cité n'est
jamais modifié, seule sa copie dans le prompt l'est.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from rag.generation.langue import Langue
from rag.models import Passage

RACINE_PROMPTS = Path(__file__).resolve().parents[3] / "prompts"
NOMS_LANGUES: dict[Langue, str] = {"fr": "French", "en": "English"}

# Toute balise ouvrante ou fermante qui ressemble à nos délimiteurs, quelle que
# soit la casse, avec espaces ou attributs : <passage>, </ Passages >, <QUESTION x="1">…
_BALISE_DELIMITEUR = re.compile(r"<\s*/?\s*(passages?|question)\b[^<>]*>", re.IGNORECASE)


@dataclass(frozen=True)
class VersionPrompt:
    version: str
    systeme: str
    gabarit: str
    empreinte_systeme: str
    empreinte_gabarit: str


@dataclass(frozen=True)
class PromptConstruit:
    systeme: str
    message: str
    version: VersionPrompt
    # Passages numérotés : numeros[i] désigne passages[i] (numérotés à partir de 1).
    passages: tuple[Passage, ...]

    @property
    def octets(self) -> int:
        """Taille du prompt en octets UTF-8 (majorant du nombre de tokens, voir D13)."""
        return len(self.systeme.encode("utf-8")) + len(self.message.encode("utf-8"))


def _empreinte(texte: str) -> str:
    return hashlib.sha256(texte.encode("utf-8")).hexdigest()


def charger_version(version: str, racine: Path = RACINE_PROMPTS) -> VersionPrompt:
    dossier = racine / version
    if not dossier.is_dir():
        raise FileNotFoundError(f"version de prompt « {version} » introuvable dans {racine}")
    systeme = (dossier / "systeme.txt").read_text(encoding="utf-8").strip()
    gabarit = (dossier / "gabarit.txt").read_text(encoding="utf-8").strip()
    for champ in ("{passages}", "{question}", "{langue_reponse}"):
        if champ not in gabarit:
            raise ValueError(f"prompt {version} : le gabarit ne contient pas {champ}")
    return VersionPrompt(version, systeme, gabarit, _empreinte(systeme), _empreinte(gabarit))


def neutraliser(texte: str) -> str:
    """Rend inoffensive toute balise qui imite nos délimiteurs (D17)."""
    return _BALISE_DELIMITEUR.sub(lambda m: m.group(0).replace("<", "‹").replace(">", "›"), texte)


def construire_prompt(
    question: str, passages: list[Passage], langue: Langue, version: VersionPrompt
) -> PromptConstruit:
    blocs = [
        f'<passage numero="{i}">\n{neutraliser(p.texte)}\n</passage>'
        for i, p in enumerate(passages, start=1)
    ]
    # str.replace et non str.format : une accolade dans un passage ou dans la
    # question (code, JSON) ne doit pas être interprétée.
    message = (
        version.gabarit.replace("{passages}", "\n".join(blocs))
        .replace("{question}", neutraliser(question.strip()))
        .replace("{langue_reponse}", NOMS_LANGUES[langue])
    )
    return PromptConstruit(version.systeme, message, version, tuple(passages))
