"""Sections d'un document Markdown et leur chemin de titres (décisions D3 et D7).

Le chemin d'une section identifie la section dans les citations et dans le jeu
de test (``sources_reference``). Règles, pour que deux personnes qui écrivent le
même chemin obtiennent exactement la même chaîne :

- titres de niveau 2 et plus, du plus général au plus précis ;
- séparateur fixe « > » entouré d'une espace : ``Imperative commands > Examples`` ;
- titres nettoyés de leur mise en forme : code (`), gras, italique, liens,
  échappements, « # » de fermeture ; espaces réduites à une seule ;
- un « > » présent dans un titre devient « › », pour que le séparateur reste
  sans ambiguïté ;
- le texte placé avant le premier titre de niveau 2 a pour chemin
  « (introduction) ». Pas le titre du document : un fichier peut contenir une
  section de niveau 2 qui porte ce même titre (« ## Persistent Volumes » dans
  persistent-volumes.md), et les deux chemins seraient identiques.

On stocke toujours le chemin complet ; ``chemin_affiche`` le raccourcit pour
l'affichage seulement.

Les titres sont cherchés hors des blocs de code : un commentaire « # ... »
dans un exemple shell n'est pas un titre.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

SEPARATEUR = " > "
INTRODUCTION = "(introduction)"

_TITRE = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)[ \t]*$")
# Toute indentation est acceptée : dans la documentation Kubernetes, beaucoup de
# blocs de code sont indentés sous un élément de liste (« 1. …\n    ```shell »).
# Avec la limite CommonMark de 3 espaces, leurs commentaires « # … » seraient
# lus comme des titres. Même règle que le nettoyage (module 1).
_OUVERTURE_CODE = re.compile(r"^[ \t]*(`{3,}|~{3,})")


def est_ouverture(ligne: str) -> str | None:
    """Délimiteur d'ouverture d'un bloc de code (```, ~~~…), ou None."""
    m = _OUVERTURE_CODE.match(ligne)
    return m.group(1) if m else None


def est_cloture(ligne: str, ouverture: str) -> bool:
    s = ligne.strip()
    return len(s) >= len(ouverture) and set(s) == {ouverture[0]}


@dataclass(frozen=True)
class Section:
    chemin: str
    # Titres du chemin, du plus général au plus précis (niveau 2 et plus).
    titres: tuple[str, ...]
    texte: str
    # Position du texte de la section dans le texte du document.
    debut: int
    exclue: bool


# --------------------------------------------------------------------------- #
# Nettoyage des titres
# --------------------------------------------------------------------------- #

_LIEN = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_GRAS = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
# Italique : le délimiteur ne doit pas être collé à une lettre de l'extérieur,
# pour ne pas toucher aux identifiants comme max_surge ou pod_name_.
_ITALIQUE = re.compile(r"(?<![\w*])([*_])(?=\S)(.+?)(?<=\S)\1(?![\w*])")
_ECHAPPEMENT = re.compile(r"\\([\\`*_{}\[\]()#+\-.!>|])")
_FERMETURE = re.compile(r"[ \t]+#+[ \t]*$")
_ANCRE = re.compile(r"[ \t]*\{#[^}]*\}[ \t]*$")


# Un caractère échappé (« \* ») est mis à l'abri dans la zone Unicode privée
# pendant le traitement de la mise en forme, pour ne pas être pris pour de l'italique.
_ABRI = 0xE000


def nettoyer_titre(titre: str) -> str:
    t = _ANCRE.sub("", titre)
    t = _FERMETURE.sub("", t)
    t = _ECHAPPEMENT.sub(lambda m: chr(_ABRI + ord(m.group(1))), t)
    t = _LIEN.sub(r"\1", t)
    t = t.replace("`", "")
    t = _GRAS.sub(r"\2", t)
    t = _ITALIQUE.sub(r"\2", t)
    t = "".join(chr(ord(c) - _ABRI) if _ABRI <= ord(c) < _ABRI + 128 else c for c in t)
    t = t.replace(">", "›")
    return " ".join(t.split())


def normaliser_titre(titre: str) -> str:
    """Forme de comparaison pour les exclusions : casse et apostrophes ignorées."""
    return titre.replace("’", "'").replace("‘", "'").casefold().strip()


def construire_chemin(titres: tuple[str, ...]) -> str:
    return SEPARATEUR.join(titres)


def chemin_affiche(chemin: str, max_caracteres: int = 60) -> str:
    """Forme courte pour l'affichage d'une citation : les deux derniers niveaux,
    précédés de « … » si des niveaux ont été omis. Le chemin stocké ne change pas."""
    if len(chemin) <= max_caracteres:
        return chemin
    parties = chemin.split(SEPARATEUR)
    if len(parties) <= 2:
        return chemin
    return "…" + SEPARATEUR + SEPARATEUR.join(parties[-2:])


# --------------------------------------------------------------------------- #
# Découpage en sections
# --------------------------------------------------------------------------- #


def decouper_sections(texte: str, sections_exclues: tuple[str, ...] = ()) -> list[Section]:
    """Sections non vides du document, dans l'ordre.

    Une section exclue (son titre ou celui d'un de ses parents figure dans
    ``sections_exclues``) est renvoyée avec ``exclue=True`` : l'appelant la
    compte mais ne la découpe pas.
    """
    exclusions = {normaliser_titre(t) for t in sections_exclues}
    pile: list[tuple[int, str]] = []
    sections: list[Section] = []

    lignes = texte.split("\n")
    position = 0  # position du début de la ligne courante dans le texte
    debut_corps = 0
    clotures_code: str | None = None

    def fermer(fin: int) -> None:
        corps = texte[debut_corps:fin]
        if not corps.strip():
            return
        titres = tuple(t for niveau, t in pile if niveau >= 2)
        chemin = construire_chemin(titres) if titres else INTRODUCTION
        exclue = any(normaliser_titre(t) in exclusions for _, t in pile)
        sections.append(Section(chemin, titres, corps, debut_corps, exclue))

    for ligne in lignes:
        fin_ligne = position + len(ligne)
        if clotures_code is not None:
            # Une clôture : même caractère, au moins aussi longue, rien d'autre.
            if est_cloture(ligne, clotures_code):
                clotures_code = None
        elif ouverture := est_ouverture(ligne):
            clotures_code = ouverture
        elif m := _TITRE.match(ligne):
            fermer(position)
            niveau, titre = len(m.group(1)), nettoyer_titre(m.group(2))
            while pile and pile[-1][0] >= niveau:
                pile.pop()
            pile.append((niveau, titre))
            debut_corps = fin_ligne + 1
        position = fin_ligne + 1
    fermer(len(texte))
    return sections


def chemins_ambigus(sections: list[Section]) -> list[str]:
    """Chemins portés par plusieurs sections du même document : une citation
    [fichier, chemin] ne désignerait pas une section unique."""
    compte = Counter(s.chemin for s in sections)
    return sorted(c for c, n in compte.items() if n > 1)
