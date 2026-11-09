"""EF05 — nettoyage du texte."""

import random

import pytest

from rag.config import NettoyageConfig
from rag.ingestion.cleaning import (
    clean_hugo_markdown,
    clean_pages,
    dehyphenate,
    normalize_characters,
    normalize_whitespace,
    remove_repeated_edges,
)
from rag.ingestion.extractors import extract_pages
from tests.conftest import EN_TETE, PIED, MakePdf, corps_page

CFG = NettoyageConfig()


# --- En-têtes et pieds de page ------------------------------------------------


def test_ef05_aucun_en_tete_repete_sur_20_pages_tirees_au_hasard(make_pdf: MakePdf) -> None:
    """Critère EF05, appliqué aux pages (les passages n'existent pas encore).

    Le PDF porte un en-tête fixe et un pied « Page n of N » sur chaque page.
    Après nettoyage, aucun des deux ne doit subsister, et le corps doit être intact.
    """
    nb_pages = 30
    pdf = make_pdf(
        pages=[corps_page(n) for n in range(1, nb_pages + 1)], en_tete=EN_TETE, pied=PIED
    )
    brutes = extract_pages(pdf, "pdf")
    assert EN_TETE in brutes[0], "pré-condition : l'en-tête est bien extrait avant nettoyage"

    pages = clean_pages(brutes, "pdf", CFG)
    for n in random.Random(7).sample(range(1, nb_pages + 1), 20):
        texte = pages[n - 1]
        assert "XR-200" not in texte
        assert f"of {nb_pages}" not in texte
        assert f"MARQUEUR-P{n:03d}" in texte
        assert "Contrôler les joints" in texte


def test_ligne_de_corps_unique_en_haut_de_page_conservee() -> None:
    titres = ["Installation", "Configuration", "Maintenance", "Dépannage", "Annexes"]
    pages = [f"En-tête commun\nChapitre {t}\ncorps {t}\nPied" for t in titres]
    nettoyees = remove_repeated_edges(pages, CFG)
    assert all("En-tête commun" not in p and "Pied" not in p for p in nettoyees)
    assert all(f"Chapitre {t}" in p for t, p in zip(titres, nettoyees, strict=True))


def test_en_tete_courant_variant_par_un_nombre_retire() -> None:
    """Limite assumée : une ligne de bord qui ne change que par ses nombres
    (« Chapitre 3 — Installation », « Rév. 12 ») est traitée comme un en-tête courant."""
    pages = [
        f"Chapitre {i // 2} — Installation\ncorps {t}\nfin {t}" for i, t in enumerate("abcdef")
    ]
    nettoyees = remove_repeated_edges(pages, CFG)
    assert all("Chapitre" not in p for p in nettoyees)


def test_pages_blanches_exclues_du_calcul_de_frequence() -> None:
    """En-tête présent sur toutes les pages qui ont du texte, mais seulement 3 pages
    sur 8 en ont : il doit quand même être reconnu."""
    pages = [
        "",
        "",
        "Entête X\ncorps a\nfin a",
        "",
        "Entête X\ncorps b\nfin b",
        "",
        "",
        "Entête X\nc\nd",
    ]
    nettoyees = remove_repeated_edges(pages, CFG)
    assert all("Entête X" not in p for p in nettoyees)


def test_phrase_repetee_au_milieu_de_page_conservee() -> None:
    """Seules les lignes de bord sont examinées."""
    pages = [
        f"titre {i}\nintro {i}\nAvertissement : couper le courant.\nfin {i}\nbas {i}"
        for i in range(6)
    ]
    nettoyees = remove_repeated_edges(pages, CFG)
    assert all("Avertissement : couper le courant." in p for p in nettoyees)


def test_numeros_de_page_isoles_retires() -> None:
    formats = ["{n}", "- {n} -", "Page {n}", "{n} / 9", "p. {n}"]
    mots = ["alpha", "bravo", "charlie", "delta"]
    for fmt in formats:
        pages = [f"Titre {m}\ncontenu {m}\n" + fmt.format(n=i + 1) for i, m in enumerate(mots)]
        nettoyees = remove_repeated_edges(pages, CFG)
        for m, p in zip(mots, nettoyees, strict=True):
            assert p == f"Titre {m}\ncontenu {m}", fmt


def test_pas_de_detection_sous_le_minimum_de_pages() -> None:
    pages = ["Entête\nA\nPied", "Entête\nB\nPied"]
    assert remove_repeated_edges(pages, CFG) == pages


def test_ligne_presente_sur_moins_de_la_moitie_des_pages_conservee() -> None:
    pages = [("Note de bas" if i < 2 else f"x{i}") + f"\ncorps {i}\nfin {i}" for i in range(6)]
    nettoyees = remove_repeated_edges(pages, CFG)
    assert "Note de bas" in nettoyees[0]


def test_nettoyage_desactivable() -> None:
    pages = [f"Entête\ncorps {i}\nPied" for i in range(5)]
    cfg = NettoyageConfig(entetes_pieds=False)
    assert all("Entête" in p for p in clean_pages(pages, "pdf", cfg))


# --- Césures ------------------------------------------------------------------


def test_cesure_recollee() -> None:
    assert dehyphenate("l'infor-\nmation est dispo-\n  nible") == "l'information est disponible"


def test_cesure_devant_majuscule_ou_chiffre_conservee() -> None:
    texte = "Jean-\nPierre et la norme ISO-\n9001"
    assert dehyphenate(texte) == texte


def test_cesures_non_appliquees_au_markdown() -> None:
    """En Markdown, un tiret en fin de ligne est volontaire (listes, tableaux)."""
    pages = clean_pages(["- élément un-\nsuite"], "md", CFG)
    assert pages == ["- élément un-\nsuite"]


# --- Caractères et espaces ----------------------------------------------------


def test_ligatures_et_espaces_speciaux() -> None:
    assert normalize_characters("conﬁguration : 12­bar") == "configuration : 12bar"


def test_accents_recomposes() -> None:
    assert normalize_characters("été") == "été"


def test_espaces_et_lignes_vides() -> None:
    texte = "  titre   \n\n\n\nmot1    mot2\t\tmot3   \n"
    assert normalize_whitespace(texte) == "  titre\n\nmot1 mot2 mot3"


def test_indentation_markdown_conservee() -> None:
    assert normalize_whitespace("* liste\n    code  indenté") == "* liste\n    code indenté"


# --- Markdown Hugo (documentation Kubernetes) -----------------------------------

HUGO = """---
title: Pods
weight: 10
---

<!-- overview -->

A Pod contains {{< glossary_tooltip text="containers" term_id="container" >}}.
{{< feature-state for_k8s_version="v1.29" state="stable" >}}
{{< feature-state feature_gate_name="SidecarContainers" >}}

{{< note >}}
Install a [container runtime](/docs/setup/runtimes/) first.
{{< /note >}}

Kubernetes v{{< skew currentVersion >}} supports this.
![schéma](/images/pod.svg)
{{< mermaid >}}
graph LR; A-->B
{{< /mermaid >}}
{{% code_sample file="pods/simple-pod.yaml" %}}

{{% heading "whatsnext" %}}
"""


def test_markdown_hugo_nettoye() -> None:
    texte = normalize_whitespace(clean_hugo_markdown(HUGO, {"currentVersion": "1.35"}))
    assert texte.startswith("# Pods\n")
    assert "weight" not in texte
    assert "overview" not in texte
    assert "A Pod contains containers." in texte
    assert "(Feature state: stable since Kubernetes v1.29)" in texte
    assert "(Feature gate: SidecarContainers)" in texte
    assert "Note:\nInstall a container runtime first." in texte
    assert "Kubernetes v1.35 supports this." in texte
    assert "## What's next" in texte
    for residu in ("{{", "}}", "](", "graph LR", "simple-pod.yaml", "/docs/setup"):
        assert residu not in texte, residu


def test_tableau_html_converti_en_texte() -> None:
    texte = (
        "<table><tr><th>Champ</th><th>Type</th></tr>\n"
        '<tr><td><code>replicas</code></td><td><a href="/x">entier</a> &gt; 0</td></tr></table>'
    )
    propre = normalize_whitespace(clean_hugo_markdown(texte))
    assert "<" not in propre.replace("> 0", "")
    assert "Champ | Type |" in propre
    assert "replicas | entier > 0 |" in propre


def test_html_et_liens_conserves_dans_les_blocs_de_code() -> None:
    bloc = "```html\n<p>exemple</p>\n[a](b)\n```"
    sortie = "```\nNAME   READY\nweb    <none>\n```"
    texte = f"Avant <em>x</em>.\n\n{bloc}\n\nSortie :\n\n{sortie}\n"
    propre = clean_hugo_markdown(texte)
    assert bloc in propre and sortie in propre
    assert "Avant x." in propre


def test_espaces_reserves_hors_code_conserves() -> None:
    """« <pod-name> » n'est pas une balise HTML connue : c'est un contenu à garder."""
    assert clean_hugo_markdown("kubectl logs <pod-name>") == "kubectl logs <pod-name>"


def test_liens_de_reference() -> None:
    texte = "Voir le [guide][g] pour plus.\n\n[g]: https://example.org/guide\n"
    assert normalize_whitespace(clean_hugo_markdown(texte)) == "Voir le guide pour plus."


def test_shortcodes_fermants_avec_espace() -> None:
    texte = "{{< note >}}\nA\n{{</ note >}}\n{{< mermaid >}}\ngraph\n{{</ mermaid >}}\nB"
    assert normalize_whitespace(clean_hugo_markdown(texte)) == "Note:\nA\n\nB"


def test_fin_de_lien_orpheline_retiree() -> None:
    texte = "des orphelins](/docs/concepts/gc/#cascading)."
    assert clean_hugo_markdown(texte) == "des orphelins."


def test_accolades_de_gabarit_conservees() -> None:
    """« {{username}} » n'est pas un shortcode Hugo (pas de < ni de %)."""
    assert clean_hugo_markdown("variables `{{username}}`") == "variables `{{username}}`"


@pytest.mark.parametrize(
    ("cle", "libelle"),
    [("whatsnext", "What's next"), ("prerequisites", "Before you begin"), ("seealso", "See also")],
)
def test_titres_hugo_du_corpus(cle: str, libelle: str) -> None:
    """Les trois clés présentes dans le corpus Kubernetes ont leur libellé officiel."""
    assert clean_hugo_markdown(f'## {{{{% heading "{cle}" %}}}}') == f"## {libelle}"


def test_titre_hugo_dans_une_ligne_de_titre_non_double() -> None:
    """Forme réelle du corpus : « ## {{% heading "whatsnext" %}} »."""
    texte = '## {{% heading "whatsnext" %}}\n\n### {{% heading "prerequisites" %}}'
    assert clean_hugo_markdown(texte) == "## What's next\n\n### Before you begin"


def test_ancres_de_titre_retirees() -> None:
    texte = (
        "## Historical context {#going-back-in-time}\n"
        "Texte avec {#accolades} hors titre.\n"
        "```\n## {#dans-le-code}\n```"
    )
    propre = clean_hugo_markdown(texte)
    assert propre.startswith("## Historical context\n")
    assert "Texte avec {#accolades} hors titre." in propre
    assert "## {#dans-le-code}" in propre


def test_glossary_tooltip_sans_text_rendu_par_term_id() -> None:
    texte = (
        'Un {{< glossary_tooltip term_id="pod" >}} et des '
        '{{< glossary_tooltip text="nœuds" term_id="node" >}}.'
    )
    assert clean_hugo_markdown(texte) == "Un pod et des nœuds."


def test_onglet_de_code_converti_en_bloc() -> None:
    texte = (
        '{{< tab name="Linux node" codelang="yaml" >}}\n---\n# Ce manifeste monte /data\n'
        "kind: Pod\n{{< /tab >}}"
    )
    propre = clean_hugo_markdown(texte)
    assert propre.strip() == "Linux node:\n```yaml\n---\n# Ce manifeste monte /data\nkind: Pod\n```"


def test_onglet_de_prose_inchange_hormis_le_shortcode() -> None:
    texte = '{{% tab name="Windows" %}}\nTexte de prose.\n{{% /tab %}}'
    assert normalize_whitespace(clean_hugo_markdown(texte)) == "Windows:\nTexte de prose."


def test_highlight_bloc_et_en_ligne() -> None:
    bloc = '{{< highlight yaml "hl_lines=6" >}}\nkind: Pod\n# commentaire\n{{< /highlight >}}'
    assert clean_hugo_markdown(bloc).strip() == "```yaml\nkind: Pod\n# commentaire\n```"
    en_ligne = (
        'La requête {{< highlight promql "hl_inline=true" >}}rate(x[5m]){{< /highlight >}} calcule.'
    )
    assert clean_hugo_markdown(en_ligne) == "La requête `rate(x[5m])` calcule."


def test_markdown_ordinaire_inchange() -> None:
    texte = "# Titre\n\nUn paragraphe avec `code` et une liste :\n\n- a\n- b"
    assert clean_hugo_markdown(texte) == texte


def test_front_matter_invalide_retire_sans_erreur() -> None:
    texte = "---\ntitle: [non fermé\n---\nCorps"
    assert clean_hugo_markdown(texte) == "Corps"


def test_hugo_desactivable() -> None:
    pages = clean_pages(["{{< note >}}x{{< /note >}}"], "md", NettoyageConfig(markdown_hugo=False))
    assert pages == ["{{< note >}}x{{< /note >}}"]


def test_nombre_de_pages_inchange() -> None:
    pages = ["a", "", "b", "   "]
    assert len(clean_pages(pages, "pdf", CFG)) == 4
