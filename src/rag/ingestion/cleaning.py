"""Nettoyage du texte extrait (EF05).

Les étapes, dans l'ordre où ``clean_pages`` les applique :

1. ``normalize_characters`` (tous formats) : ligatures typographiques (« ﬁ » → « fi »),
   espaces insécables, tirets conditionnels, caractères invisibles.
2. ``clean_hugo_markdown`` (Markdown, si activé) : front matter, commentaires,
   shortcodes Hugo, liens, balises HTML (hors blocs de code). Pensé pour le
   corpus Kubernetes ; sur un Markdown ordinaire, seuls liens et HTML changent.
3. ``remove_repeated_edges`` (PDF) : en-têtes et pieds de page répétés, numéros de page.
4. ``dehyphenate`` (PDF) : césures de fin de ligne.
5. ``normalize_whitespace`` (tous formats) : espaces multiples, lignes vides en excès.

Le nettoyage travaille sur la liste des pages et renvoie une liste de même
longueur : il ne fusionne ni ne supprime jamais de page.
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter
from collections.abc import Callable

import yaml

from rag.config import NettoyageConfig
from rag.ingestion.validation import Format

# À incrémenter à chaque changement du code qui modifie le texte nettoyé. Un
# document nettoyé par une version antérieure est retraité au prochain import
# (sinon, ayant la même empreinte, il serait vu comme un doublon et garderait
# l'ancien texte). Historique :
#   1 — version initiale du module 1
#   2 — titres : « ## ## What's next » et ancres {#...} corrigés ;
#       glossary_tooltip sans attribut text rendu par son term_id
#   3 — code placé dans {{< tab codelang=… >}} ou {{< highlight … >}} converti en
#       bloc de code (ses commentaires « # … » étaient lus comme des titres)
#   4 — titre {{% heading "seealso" %}} rendu « See also » (et non « Seealso »)
VERSION_NETTOYAGE = 4

# --------------------------------------------------------------------------- #
# 1. Caractères
# --------------------------------------------------------------------------- #

_REMPLACEMENTS = str.maketrans(
    {
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ﬅ": "st",
        "ﬆ": "st",
        " ": " ",  # espace insécable
        " ": " ",  # espace fine insécable (typographie française)
        " ": " ",
        " ": " ",
        "­": "",  # tiret conditionnel (césure invisible)
        "​": "",  # espace de largeur nulle
        "﻿": "",  # BOM égaré
    }
)


def normalize_characters(texte: str) -> str:
    # NFC recompose « e + accent combinant » en « é » sans toucher au reste.
    # NFKC irait plus loin mais transformerait aussi « m² » en « m2 ».
    return unicodedata.normalize("NFC", texte).translate(_REMPLACEMENTS)


# --------------------------------------------------------------------------- #
# 2. Markdown Hugo
# --------------------------------------------------------------------------- #

_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---[ \t]*(?:\n|\Z)", re.DOTALL)
_COMMENTAIRE_HTML = re.compile(r"<!--.*?-->", re.DOTALL)
_MERMAID = re.compile(r"\{\{<\s*mermaid\s*>\}\}.*?\{\{<\s*/\s*mermaid\s*>\}\}", re.DOTALL)
# « {{< note >}} », « {{</ note >}} », « {{% heading "x" %}} »…
# Code dans un shortcode : Hugo le met en forme comme du code, mais une fois le
# shortcode retiré il deviendrait du texte nu, et un commentaire YAML ou shell
# « # … » serait pris pour un titre. On le convertit en vrai bloc de code.
_TAB_CODE = re.compile(r"\{\{<\s*tab\s+([^>]*?)>\}\}(.*?)\{\{<\s*/\s*tab\s*>\}\}", re.DOTALL)
_HIGHLIGHT = re.compile(
    r"\{\{<\s*highlight\s+([\w-]+)([^>]*?)>\}\}(.*?)\{\{<\s*/\s*highlight\s*>\}\}", re.DOTALL
)
_SHORTCODE = re.compile(r"\{\{[<%]\s*(/?)\s*([\w-]+)(.*?)\s*[>%]\}\}", re.DOTALL)
_ATTRIBUT = re.compile(r'([\w-]+)\s*=\s*"([^"]*)"')
_ARGUMENT_SEUL = re.compile(r'"([^"]*)"|(\S+)')
_IMAGE_MD = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LIEN_MD = re.compile(r"\[([^\]]+)\]\([^)]*\)")

_DEBUT_TITRE = re.compile(r"[ \t]*#{1,6}[ \t]+")
# Ancre Hugo en fin de titre : « ## Historical context {#going-back-in-time} ».
_ANCRE_TITRE = re.compile(r"^([ \t]*#{1,6}[ \t].*?)[ \t]*\{#[^}\n]*\}[ \t]*$", re.MULTILINE)

_ENCADRES = {"note": "Note:", "caution": "Caution:", "warning": "Warning:"}
_TITRES_HUGO = {
    "prerequisites": "Before you begin",
    "whatsnext": "What's next",
    "objectives": "Objectives",
    "cleanup": "Clean up",
    "synopsis": "Synopsis",
    "options": "Options",
    "seealso": "See also",
}


def _attributs(arguments: str) -> dict[str, str]:
    return dict(_ATTRIBUT.findall(arguments))


def _premier_argument(arguments: str) -> str:
    m = _ARGUMENT_SEUL.search(arguments)
    if not m:
        return ""
    return m.group(1) if m.group(1) is not None else m.group(2)


def _bloc_code(langage: str, contenu: str) -> str:
    return f"\n```{langage}\n{contenu.strip(chr(10))}\n```\n"


def _remplacer_tab_code(m: re.Match[str]) -> str:
    attrs = _attributs(m.group(1))
    if "codelang" not in attrs:
        return m.group(0)  # onglet de prose : traité par la règle générale
    titre = f"{attrs['name']}:" if "name" in attrs else ""
    return titre + _bloc_code(attrs["codelang"], m.group(2))


def _remplacer_highlight(m: re.Match[str]) -> str:
    langage, options, contenu = m.group(1), m.group(2), m.group(3)
    if "hl_inline=true" in options:
        return f"`{contenu.strip()}`"
    return _bloc_code(langage, contenu)


def _remplacer_shortcode(m: re.Match[str], variables: dict[str, str]) -> str:
    fermant, nom, arguments = m.group(1), m.group(2), m.group(3)
    if fermant:
        return ""
    attrs = _attributs(arguments)
    if nom == "glossary_tooltip":
        # Sans « text », Hugo affiche le nom du terme ; son term_id (« pod »,
        # « service-account ») en est la meilleure approximation disponible.
        return attrs.get("text") or attrs.get("term_id", "")
    if nom in _ENCADRES:
        return _ENCADRES[nom]
    if nom == "feature-state":
        if "state" in attrs and "for_k8s_version" in attrs:
            version = attrs["for_k8s_version"].lstrip("v")
            return f"(Feature state: {attrs['state']} since Kubernetes v{version})"
        if "feature_gate_name" in attrs:
            return f"(Feature gate: {attrs['feature_gate_name']})"
        return ""
    if nom == "heading":
        cle = _premier_argument(arguments)
        libelle = _TITRES_HUGO.get(cle, cle.capitalize())
        # Dans la source, le shortcode suit presque toujours un « ## » : on ne rend
        # alors que le libellé, sinon on obtiendrait « ## ## What's next ».
        debut_ligne = m.string.rfind("\n", 0, m.start()) + 1
        if _DEBUT_TITRE.fullmatch(m.string[debut_ligne : m.start()]):
            return libelle
        return f"## {libelle}"
    if nom in ("skew", "param"):
        return variables.get(_premier_argument(arguments), "")
    if nom == "tab" and "name" in attrs:
        return f"{attrs['name']}:"
    # Tout autre shortcode (code_sample, figure, glossary_definition...) renvoie à
    # un contenu externe que l'on n'a pas : on le retire.
    return ""


def clean_hugo_markdown(texte: str, variables: dict[str, str] | None = None) -> str:
    variables = variables or {}

    m = _FRONT_MATTER.match(texte)
    if m:
        titre = ""
        try:
            meta = yaml.safe_load(m.group(1))
            if isinstance(meta, dict) and meta.get("title"):
                titre = str(meta["title"]).strip()
        except yaml.YAMLError:
            pass
        # Le titre devient un vrai titre Markdown : le découpage par titres (EF07)
        # et les citations pourront s'en servir.
        texte = (f"# {titre}\n\n" if titre else "") + texte[m.end() :]

    texte = _COMMENTAIRE_HTML.sub("", texte)
    texte = _MERMAID.sub("", texte)
    texte = _TAB_CODE.sub(_remplacer_tab_code, texte)
    texte = _HIGHLIGHT.sub(_remplacer_highlight, texte)
    texte = _SHORTCODE.sub(lambda mm: _remplacer_shortcode(mm, variables), texte)
    # Le reste ne s'applique pas dans les blocs de code ```...``` : un exemple
    # HTML ou une sortie « <none> » de kubectl y sont du vrai contenu.
    return _hors_blocs_de_code(texte, _nettoyer_prose_markdown)


_BLOC_CODE = re.compile(r"^[ \t]*(```|~~~).*?^[ \t]*\1[ \t]*$", re.DOTALL | re.MULTILINE)
_DEFINITION_REF = re.compile(r"^[ \t]*\[[^\]]+\]:[ \t]*\S+.*$", re.MULTILINE)
_LIEN_REF = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
# Fin de lien sans crochet ouvrant (erreur présente dans la source) : « texte](/docs/x) ».
_LIEN_ORPHELIN = re.compile(r"\]\((?:/|https?://)[^)\s]*\)")
# Liste fermée de balises HTML : un texte comme « <pod-name> » ou « <none> »
# n'est pas une balise connue et reste intact.
_BALISES = (
    r"a|abbr|b|blockquote|br|caption|code|dd|del|details|div|dl|dt|em|figcaption|figure|"
    r"h[1-6]|hr|i|iframe|img|ins|kbd|li|nav|ol|p|pre|s|section|small|source|span|strong|"
    r"sub|summary|sup|table|tbody|td|tfoot|th|thead|tr|u|ul|video"
)
_FIN_CELLULE = re.compile(r"</(td|th)\s*>", re.IGNORECASE)
_FIN_BLOC = re.compile(r"<br\s*/?>|</(p|li|tr|div|dt|dd|h[1-6]|caption|summary)\s*>", re.IGNORECASE)
_BALISE = re.compile(rf"</?({_BALISES})(\s[^<>]*)?/?>", re.IGNORECASE)


def _hors_blocs_de_code(texte: str, fonction: Callable[[str], str]) -> str:
    morceaux, debut = [], 0
    for bloc in _BLOC_CODE.finditer(texte):
        morceaux.append(fonction(texte[debut : bloc.start()]))
        morceaux.append(bloc.group(0))
        debut = bloc.end()
    morceaux.append(fonction(texte[debut:]))
    return "".join(morceaux)


def _nettoyer_prose_markdown(texte: str) -> str:
    # Les titres deviennent des citations [fichier, section] : ils doivent être propres.
    texte = _ANCRE_TITRE.sub(r"\1", texte)
    texte = _IMAGE_MD.sub("", texte)
    # On garde le texte du lien, pas l'URL : l'URL n'aide pas la recherche par sens
    # et pollue les passages.
    texte = _LIEN_MD.sub(r"\1", texte)
    texte = _LIEN_REF.sub(r"\1", texte)
    texte = _LIEN_ORPHELIN.sub("", texte)
    texte = _DEFINITION_REF.sub("", texte)
    # Tableaux et listes écrits en HTML brut : on garde le texte et une structure
    # minimale (cellules séparées par « | », lignes par des retours).
    texte = _FIN_CELLULE.sub(" | ", texte)
    texte = _FIN_BLOC.sub("\n", texte)
    texte = _BALISE.sub("", texte)
    return html.unescape(texte)


# --------------------------------------------------------------------------- #
# 3. En-têtes et pieds de page répétés (PDF)
# --------------------------------------------------------------------------- #

_CHIFFRES = re.compile(r"\d+")
_ESPACES = re.compile(r"\s+")
# Clé normalisée d'une ligne réduite à un numéro de page : « 12 », « - 12 - »,
# « Page 3 of 14 », « 3 / 14 », « p. 7 ».
_NUMERO_PAGE = re.compile(r"^(page|p\.)?\s*[-–—]?\s*#\s*[-–—]?\s*((/|of|sur|de)\s*#)?$")


def _cle_ligne(ligne: str) -> str:
    """Forme normalisée qui rend identiques « Page 3 of 14 » et « Page 4 of 14 »."""
    cle = _CHIFFRES.sub("#", ligne.lower())
    return _ESPACES.sub(" ", cle).strip()


def _indices_bord(lignes: list[str], k: int) -> set[int]:
    non_vides = [i for i, ligne in enumerate(lignes) if ligne.strip()]
    return set(non_vides[:k]) | set(non_vides[-k:])


def remove_repeated_edges(pages: list[str], cfg: NettoyageConfig) -> list[str]:
    """Retire les lignes de bord (haut/bas) qui reviennent sur une part des pages
    au moins égale à ``seuil_repetition``, ainsi que les numéros de page isolés.

    On ne regarde que les ``lignes_bord`` premières et dernières lignes non vides
    de chaque page : une phrase du corps de texte qui se répète n'est pas touchée.
    Les chiffres sont neutralisés pour reconnaître « Page 3 » et « Page 4 ».
    Conséquence assumée : une ligne de bord qui ne change que par ses nombres
    (« Chapitre 3 — Installation ») est traitée comme un en-tête courant.

    La fréquence est calculée sur les pages qui contiennent du texte : des pages
    blanches (schémas, pages numérisées) ne doivent pas masquer un en-tête.
    """
    n = sum(1 for p in pages if p.strip())
    if n < cfg.min_pages_detection:
        return pages

    lignes_par_page = [p.split("\n") for p in pages]
    bords = [_indices_bord(lignes, cfg.lignes_bord) for lignes in lignes_par_page]

    compte: Counter[str] = Counter()
    for lignes, idx in zip(lignes_par_page, bords, strict=True):
        compte.update({_cle_ligne(lignes[i]) for i in idx})  # set : une fois par page
    repetees = {cle for cle, c in compte.items() if cle and c / n >= cfg.seuil_repetition}

    resultat = []
    for lignes, idx in zip(lignes_par_page, bords, strict=True):
        gardees = []
        for i, ligne in enumerate(lignes):
            if i in idx:
                cle = _cle_ligne(ligne)
                if cle in repetees or _NUMERO_PAGE.match(cle):
                    continue
            gardees.append(ligne)
        resultat.append("\n".join(gardees))
    return resultat


# --------------------------------------------------------------------------- #
# 4. Césures (PDF)
# --------------------------------------------------------------------------- #

# Lettre + tiret en fin de ligne, puis une lettre minuscule au début de la ligne
# suivante. « infor-\nmation » → « information ».
# On ne recolle PAS si la suite commence par une majuscule (« Jean-\nPierre »,
# « Kubernetes-\nNative ») ni par un chiffre (« ISO-\n9001 »).
# Limite connue : un vrai trait d'union tombé en fin de ligne devant une
# minuscule (« state-\nof-the-art ») perd son tiret.
_CESURE = re.compile(r"([^\W\d_])-\n[ \t]*([^\W\d_])")


def dehyphenate(texte: str) -> str:
    def _recoller(m: re.Match[str]) -> str:
        if m.group(2).islower():
            return m.group(1) + m.group(2)
        return m.group(0)

    return _CESURE.sub(_recoller, texte)


# --------------------------------------------------------------------------- #
# 5. Espaces
# --------------------------------------------------------------------------- #

_ESPACES_INTERNES = re.compile(r"[ \t]+")
_LIGNES_VIDES = re.compile(r"\n{3,}")


def normalize_whitespace(texte: str) -> str:
    lignes = []
    for ligne in texte.split("\n"):
        # On garde l'indentation de début de ligne (listes et code en Markdown)
        # et on réduit les espaces multiples dans le reste de la ligne.
        contenu = ligne.lstrip(" \t")
        indentation = ligne[: len(ligne) - len(contenu)].replace("\t", "    ")
        contenu = _ESPACES_INTERNES.sub(" ", contenu).rstrip()
        lignes.append(indentation + contenu if contenu else "")
    return _LIGNES_VIDES.sub("\n\n", "\n".join(lignes)).strip("\n")


# --------------------------------------------------------------------------- #
# Enchaînement
# --------------------------------------------------------------------------- #


def clean_pages(pages: list[str], fmt: Format, cfg: NettoyageConfig) -> list[str]:
    pages = [normalize_characters(p) for p in pages]
    if fmt == "md" and cfg.markdown_hugo:
        pages = [clean_hugo_markdown(p, cfg.variables_hugo) for p in pages]
    if fmt == "pdf":
        # Les en-têtes avant les césures : un en-tête qui finit par « - » ne doit
        # pas être recollé à la première ligne du corps.
        if cfg.entetes_pieds:
            pages = remove_repeated_edges(pages, cfg)
        if cfg.cesures:
            pages = [dehyphenate(p) for p in pages]
    return [normalize_whitespace(p) for p in pages]
