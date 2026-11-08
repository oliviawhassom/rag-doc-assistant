"""Découpage d'un texte en passages (EF06, EF07, décision D5).

Les deux fonctions renvoient des intervalles (début, fin) de caractères dans le
texte reçu ; l'appelant en tire les passages par simple tranche. Elles
s'appliquent à UNE section : un passage ne franchit jamais une frontière de
section (D5), donc la citation [fichier, section] est toujours exacte.

``fenetres`` — stratégie « taille fixe » : fenêtres de ``taille`` tokens, qui se
chevauchent de ``chevauchement`` tokens. Si la dernière fenêtre n'apporte que
quelques tokens nouveaux (moins de 10 % de la taille), elle est absorbée par la
précédente, à condition que celle-ci ne dépasse pas taille + 10 % : on évite
des passages de 3 tokens sans sortir de la tolérance d'EF06.

``par_blocs`` — stratégie « par titres » : la section est découpée en blocs
(paragraphes séparés par une ligne vide ; un bloc de code ```…``` forme un seul
bloc, même s'il contient des lignes vides), puis les blocs consécutifs sont
regroupés tant que le groupe tient dans ``taille`` tokens. Un bloc plus long que
la taille, code compris, est découpé en fenêtres fixes. Pas de chevauchement
entre groupes : les coupes tombent entre deux paragraphes.

Conséquence voulue : une section qui tient dans ``taille`` tokens donne un seul
passage avec les deux stratégies. Elles ne diffèrent que sur les sections plus
longues que la taille (proportion mesurée dans les statistiques du découpage).
"""

from __future__ import annotations

from typing import Protocol

from rag.chunking.sections import est_cloture, est_ouverture

Intervalle = tuple[int, int]

TOLERANCE = 0.10  # EF06 : taille respectée à 10 % près


class Compteur(Protocol):
    def positions(self, texte: str) -> list[tuple[int, int]]: ...
    def compter(self, texte: str) -> int: ...


def fenetres(texte: str, tok: Compteur, taille: int, chevauchement: int) -> list[Intervalle]:
    positions = tok.positions(texte)
    n = len(positions)
    if n == 0:
        return []
    if n <= taille:
        return [(positions[0][0], positions[-1][1])]

    pas = taille - chevauchement
    bornes: list[tuple[int, int]] = []
    debut = 0
    while True:
        fin = min(debut + taille, n)
        bornes.append((debut, fin))
        if fin == n:
            break
        debut += pas

    if len(bornes) >= 2:
        (deb_prec, fin_prec), (_, fin_der) = bornes[-2], bornes[-1]
        nouveaux = fin_der - fin_prec
        if nouveaux < TOLERANCE * taille and fin_der - deb_prec <= (1 + TOLERANCE) * taille:
            bornes[-2:] = [(deb_prec, fin_der)]

    return [(positions[d][0], positions[f - 1][1]) for d, f in bornes]


def blocs(texte: str) -> list[Intervalle]:
    """Paragraphes et blocs de code, dans l'ordre. Un bloc de code n'est jamais coupé."""
    resultat: list[Intervalle] = []
    debut: int | None = None
    fin = 0
    cloture: str | None = None
    position = 0

    def fermer() -> None:
        nonlocal debut
        if debut is not None:
            resultat.append((debut, fin))
            debut = None

    for ligne in texte.split("\n"):
        fin_ligne = position + len(ligne)
        if cloture is not None:
            fin = fin_ligne
            if est_cloture(ligne, cloture):
                cloture = None
                fermer()
        elif ouverture := est_ouverture(ligne):
            fermer()
            debut, fin, cloture = position, fin_ligne, ouverture
        elif not ligne.strip():
            fermer()
        else:
            if debut is None:
                debut = position
            fin = fin_ligne
        position = fin_ligne + 1
    fermer()  # y compris un bloc de code jamais refermé
    return resultat


def par_blocs(texte: str, tok: Compteur, taille: int, chevauchement: int) -> list[Intervalle]:
    groupes: list[Intervalle] = []
    courant: Intervalle | None = None

    for debut, fin in blocs(texte):
        if tok.compter(texte[debut:fin]) > taille:
            if courant is not None:
                groupes.append(courant)
                courant = None
            groupes.extend(
                (debut + d, debut + f)
                for d, f in fenetres(texte[debut:fin], tok, taille, chevauchement)
            )
        elif courant is None:
            courant = (debut, fin)
        elif tok.compter(texte[courant[0] : fin]) <= taille:
            courant = (courant[0], fin)
        else:
            groupes.append(courant)
            courant = (debut, fin)

    if courant is not None:
        groupes.append(courant)
    return groupes
