"""D3 et D7 — sections Markdown, chemins de titres, exclusions."""

import pytest

from rag.chunking.sections import (
    INTRODUCTION,
    chemin_affiche,
    chemins_ambigus,
    decouper_sections,
    nettoyer_titre,
)

# --- Nettoyage des titres : deux écritures du même titre → même chaîne ---------


@pytest.mark.parametrize(
    ("brut", "attendu"),
    [
        ("`List` gRPC endpoint", "List gRPC endpoint"),
        ("_Equality-based_ requirement", "Equality-based requirement"),
        ("**Pod** lifecycle", "Pod lifecycle"),
        ("*Pod* __lifecycle__", "Pod lifecycle"),
        ("The [Pod](/docs/pods/) API", "The Pod API"),
        ("Setting `max_surge` and pod_name_", "Setting max_surge and pod_name_"),
        (r"Escaped \*star\* and \_underscore\_", "Escaped *star* and _underscore_"),
        ("Closing hashes ##", "Closing hashes"),
        ("Anchor {#grpc-endpoint-list}", "Anchor"),
        ("A  >  B", "A › B"),
        ("  spaces   inside  ", "spaces inside"),
    ],
)
def test_nettoyer_titre(brut: str, attendu: str) -> None:
    assert nettoyer_titre(brut) == attendu


def test_deux_ecritures_du_meme_titre_donnent_le_meme_chemin() -> None:
    a = decouper_sections("## The `Pod` API\n### _Examples_\ntexte")
    b = decouper_sections("##   The Pod API ##\n### Examples {#ex}\ntexte")
    assert a[0].chemin == b[0].chemin == "The Pod API > Examples"


# --- Chemins ------------------------------------------------------------------

DOC = """# Pods

Intro du document.

## Pod lifecycle

Corps du cycle de vie.

### Pod phase

Corps des phases.

#### Pending

Corps pending.

### Container states

Corps des états.

## Init containers

Corps init.
"""


def test_chemins_complets_a_partir_du_niveau_2() -> None:
    chemins = [s.chemin for s in decouper_sections(DOC)]
    assert chemins == [
        INTRODUCTION,
        "Pod lifecycle",
        "Pod lifecycle > Pod phase",
        "Pod lifecycle > Pod phase > Pending",
        "Pod lifecycle > Container states",
        "Init containers",
    ]


def test_texte_de_section_sans_la_ligne_de_titre() -> None:
    sections = decouper_sections(DOC)
    assert sections[2].texte.strip() == "Corps des phases."
    assert all("#" not in s.texte for s in sections)


def test_position_des_sections_dans_le_texte() -> None:
    for s in decouper_sections(DOC):
        assert DOC[s.debut : s.debut + len(s.texte)] == s.texte


def test_introduction_jamais_confondue_avec_une_section_du_meme_nom() -> None:
    """Cas réel : persistent-volumes.md a pour titre « Persistent Volumes » et
    contient une section « ## Persistent Volumes »."""
    texte = "# Persistent Volumes\n\nIntro.\n\n## Persistent Volumes\n\nCorps."
    sections = decouper_sections(texte)
    assert [s.chemin for s in sections] == [INTRODUCTION, "Persistent Volumes"]
    assert chemins_ambigus(sections) == []


def test_niveau_saute() -> None:
    sections = decouper_sections("## A\n\n#### Profond\n\ntexte\n\n### B\n\ntexte")
    assert [s.chemin for s in sections] == ["A > Profond", "A > B"]


def test_section_vide_ignoree() -> None:
    sections = decouper_sections("## Parent\n\n### Enfant\n\ntexte")
    assert [s.chemin for s in sections] == ["Parent > Enfant"]


@pytest.mark.parametrize(
    "bloc",
    [
        "```bash\n# Créer le volume\nkubectl apply\n```",
        "~~~\n# commentaire\n~~~",
        "1. Étape :\n\n    ```shell\n    # commentaire indenté\n    ```",
        "````\n```\n# commentaire dans un bloc à 4 accents\n```\n````",
    ],
)
def test_commentaires_dans_un_bloc_de_code_ne_sont_pas_des_titres(bloc: str) -> None:
    texte = f"## Section\n\n{bloc}\n\nSuite."
    sections = decouper_sections(texte)
    assert [s.chemin for s in sections] == ["Section"]
    assert "commentaire" in sections[0].texte or "Créer" in sections[0].texte


def test_chemins_ambigus_detectes() -> None:
    texte = "## Exemples\n\na\n\n## Autre\n\nb\n\n## Exemples\n\nc"
    assert chemins_ambigus(decouper_sections(texte)) == ["Exemples"]


def test_meme_titre_sous_deux_parents_non_ambigu() -> None:
    """Cas réel de object-management.md : « Examples » sous deux parents."""
    texte = "## Imperative\n\n### Examples\n\na\n\n## Declarative\n\n### Examples\n\nb"
    sections = decouper_sections(texte)
    assert chemins_ambigus(sections) == []
    assert {s.chemin for s in sections} == {"Imperative > Examples", "Declarative > Examples"}


# --- Exclusions (D7) ----------------------------------------------------------


def test_section_exclue_et_ses_sous_sections() -> None:
    texte = "## Usage\n\na\n\n## What's next\n\nliens\n\n### More\n\nautres liens"
    sections = decouper_sections(texte, ("What's next",))
    assert [(s.chemin, s.exclue) for s in sections] == [
        ("Usage", False),
        ("What's next", True),
        ("What's next > More", True),
    ]


@pytest.mark.parametrize("titre", ["What’s next", "WHAT'S NEXT", "what's next"])
def test_exclusion_insensible_a_la_casse_et_a_l_apostrophe(titre: str) -> None:
    sections = decouper_sections(f"## {titre}\n\nliens", ("What's next",))
    assert sections[0].exclue


# --- Affichage ----------------------------------------------------------------


def test_chemin_affiche_court_inchange() -> None:
    assert chemin_affiche("Pod lifecycle > Pod phase") == "Pod lifecycle > Pod phase"


def test_chemin_affiche_long_raccourci_aux_deux_derniers_niveaux() -> None:
    chemin = "Kubernetes object management > Imperative object configuration > Examples"
    assert chemin_affiche(chemin, max_caracteres=40) == (
        "… > Imperative object configuration > Examples"
    )
