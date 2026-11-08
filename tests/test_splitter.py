"""EF06, EF07, D5 — fenêtres de taille fixe et regroupement par paragraphes.

Avec le tokenizer factice, un mot = un token : les tailles se vérifient à la main.
"""

import pytest

from rag.chunking.splitter import blocs, fenetres, par_blocs
from rag.chunking.tokenizer import TokenizerTexte
from tests.conftest import mots


def _textes(texte: str, intervalles: list[tuple[int, int]]) -> list[str]:
    return [texte[d:f] for d, f in intervalles]


# --- Taille fixe (EF06) -------------------------------------------------------


def test_texte_court_un_seul_passage(tok: TokenizerTexte) -> None:
    texte = mots(50)
    assert _textes(texte, fenetres(texte, tok, 100, 10)) == [texte]


def test_texte_vide(tok: TokenizerTexte) -> None:
    assert fenetres("   ", tok, 100, 10) == []


def test_fenetres_taille_et_chevauchement_exacts(tok: TokenizerTexte) -> None:
    texte = mots(250)
    passages = _textes(texte, fenetres(texte, tok, 100, 20))
    assert [tok.compter(p) for p in passages] == [100, 100, 90]
    # chevauchement : les 20 derniers mots d'un passage ouvrent le suivant
    assert passages[0].split()[-20:] == passages[1].split()[:20]
    assert passages[1].split()[-20:] == passages[2].split()[:20]


def test_ef06_tolerance_sur_un_long_texte(tok: TokenizerTexte) -> None:
    """Critère EF06 (lecture D5) : aucun passage au-dessus de taille + 10 % ;
    tous les passages sauf le dernier de la section au moins à taille − 10 %."""
    texte = mots(5_000)
    for taille, chevauchement in [(256, 32), (512, 64), (1024, 128), (100, 0)]:
        comptes = [
            tok.compter(p) for p in _textes(texte, fenetres(texte, tok, taille, chevauchement))
        ]
        assert max(comptes) <= 1.1 * taille
        assert min(comptes[:-1]) >= 0.9 * taille


def test_queue_minuscule_absorbee(tok: TokenizerTexte) -> None:
    """205 mots, taille 100, sans chevauchement : la 3e fenêtre n'aurait que 5 mots ;
    elle est absorbée par la 2e (105 mots ≤ 110)."""
    texte = mots(205)
    comptes = [tok.compter(p) for p in _textes(texte, fenetres(texte, tok, 100, 0))]
    assert comptes == [100, 105]


def test_queue_non_absorbee_si_tolerance_depassee(tok: TokenizerTexte) -> None:
    """Avec chevauchement, absorber la queue ferait dépasser 110 : on la garde."""
    texte = mots(185)
    comptes = [tok.compter(p) for p in _textes(texte, fenetres(texte, tok, 100, 20))]
    assert max(comptes) <= 110
    assert sum(comptes) - 20 * (len(comptes) - 1) == 185


def test_tout_le_texte_est_couvert(tok: TokenizerTexte) -> None:
    texte = mots(1_000)
    vus = set()
    for p in _textes(texte, fenetres(texte, tok, 128, 16)):
        vus.update(p.split())
    assert vus == set(texte.split())


def test_passage_est_une_tranche_exacte_du_texte(tok: TokenizerTexte) -> None:
    """On découpe aux positions des tokens, sans reconstruire de texte."""
    texte = "Ponctuation, accents : « été » ; code `kubectl get pods` !  " * 40
    for d, f in fenetres(texte, tok, 50, 5):
        assert texte[d:f] == texte[d:f].strip()
        assert texte[d:f] in texte


# --- Blocs ----------------------------------------------------------------------

SECTION = """Premier paragraphe
sur deux lignes.

Deuxième paragraphe.

```yaml
apiVersion: v1

kind: Pod
```

Dernier paragraphe."""


def test_blocs_paragraphes_et_code(tok: TokenizerTexte) -> None:
    morceaux = _textes(SECTION, blocs(SECTION))
    assert morceaux == [
        "Premier paragraphe\nsur deux lignes.",
        "Deuxième paragraphe.",
        "```yaml\napiVersion: v1\n\nkind: Pod\n```",
        "Dernier paragraphe.",
    ]


def test_bloc_de_code_non_ferme_va_jusqu_au_bout() -> None:
    texte = "Intro.\n\n```\nligne 1\n\nligne 2"
    assert _textes(texte, blocs(texte))[-1] == "```\nligne 1\n\nligne 2"


# --- Par titres (EF07, D5) ------------------------------------------------------


def test_section_courte_un_seul_passage_dans_les_deux_strategies(tok: TokenizerTexte) -> None:
    assert len(par_blocs(SECTION, tok, 256, 32)) == 1
    assert len(fenetres(SECTION, tok, 256, 32)) == 1


def test_regroupement_jusqu_a_la_taille(tok: TokenizerTexte) -> None:
    paragraphes = [mots(30, f"p{i}_") for i in range(10)]  # 10 × 30 mots
    texte = "\n\n".join(paragraphes)
    passages = _textes(texte, par_blocs(texte, tok, 100, 10))
    # 3 paragraphes (90 mots) tiennent dans 100 ; un 4e ferait 120
    assert [tok.compter(p) for p in passages] == [90, 90, 90, 30]
    # les coupes tombent entre deux paragraphes
    for p in passages:
        assert all(p.count(f"p{i}_0 ") in (0, 1) for i in range(10))
        assert p.split("\n\n")[0] in paragraphes


def test_bloc_de_code_jamais_coupe_s_il_tient(tok: TokenizerTexte) -> None:
    code = "```\n" + "\n".join(mots(8, f"l{i}_") for i in range(10)) + "\n```"  # ~80 tokens
    texte = f"{mots(60)}\n\n{code}\n\n{mots(20, 'f')}"
    passages = _textes(texte, par_blocs(texte, tok, 100, 10))
    assert sum(code in p for p in passages) == 1


def test_bloc_de_code_plus_long_que_la_taille_en_fenetres_fixes(tok: TokenizerTexte) -> None:
    code = "```\n" + mots(300, "c") + "\n```"
    texte = f"Intro courte.\n\n{code}\n\nConclusion."
    passages = _textes(texte, par_blocs(texte, tok, 100, 10))
    assert passages[0] == "Intro courte."
    assert passages[-1] == "Conclusion."
    milieu = [tok.compter(p) for p in passages[1:-1]]
    assert len(milieu) >= 3
    assert max(milieu) <= 110


@pytest.mark.parametrize("taille", [64, 128, 256])
def test_par_blocs_jamais_au_dessus_de_la_tolerance(tok: TokenizerTexte, taille: int) -> None:
    texte = "\n\n".join(mots(n, f"b{n}_") for n in [5, 300, 40, 40, 90, 12, 700, 3])
    for p in _textes(texte, par_blocs(texte, tok, taille, 8)):
        assert tok.compter(p) <= 1.1 * taille
