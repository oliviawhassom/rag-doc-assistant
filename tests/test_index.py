"""EF08, EF09, EF11, D9, D11, D12 — index vectoriel, indexation, recherche.

L'encodeur factice remplace le vrai modèle ; le calcul réel des vecteurs est
vérifié par tests/test_embeddings_reels.py (tests « slow »).
"""

import logging
import time
from pathlib import Path

import numpy as np
import pytest

from rag.config import Config, DecoupageConfig, EmbeddingsConfig, StockageConfig
from rag.index import (
    IndexPerime,
    IndexVectoriel,
    ModeleIndisponible,
    MoteurRecherche,
    charger_index_a_jour,
    indexer,
    resoudre_revision,
)
from rag.index.indexation import calculer_empreinte, parametres_encodage, textes_encodes
from rag.models import Passage
from tests.conftest import EncodeurFactice

REVISION = "a" * 40


def _passage(i: int, texte: str, section: str | None = "Section") -> Passage:
    return Passage(
        passage_id=f"doc_{i:04d}",
        document_id="doc",
        page_debut=1,
        page_fin=1,
        titre_section=section,
        texte=texte,
        nb_tokens=len(texte.split()),
        config_decoupage="cfg",
    )


PASSAGES = [
    _passage(0, "A Pod is the smallest deployable unit; create a Pod with kubectl.", "Pods"),
    _passage(1, "A Service exposes an application running on a set of Pods.", "Services"),
    _passage(2, "A Secret stores sensitive data such as passwords and tokens.", "Secrets"),
    _passage(3, "Nodes are the machines that run your workloads.", "Nodes"),
]


def _config(tmp_path: Path, **embeddings: object) -> Config:
    return Config(
        version="test",
        stockage=StockageConfig(dossier=tmp_path / "store"),
        embeddings=EmbeddingsConfig(**embeddings),  # type: ignore[arg-type]
    )


def _jamais_appelee() -> EncodeurFactice:
    raise AssertionError("le modèle ne doit pas être chargé : l'index est à jour")


# --- Index vectoriel (D9) ---------------------------------------------------------------------


def _vecteurs(*lignes: list[float]) -> np.ndarray:
    v = np.array(lignes, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_recherche_exacte_ordre_et_scores() -> None:
    index = IndexVectoriel.construire(["a", "b", "c"], _vecteurs([1, 0], [0.6, 0.8], [0, 1]))
    resultats = index.rechercher(_vecteurs([1, 0])[0], k=3)
    assert [pid for pid, _ in resultats] == ["a", "b", "c"]
    assert [round(s, 4) for _, s in resultats] == [1.0, 0.6, 0.0]


def test_ex_aequo_departages_par_passage_id() -> None:
    """D9 : à score identique, l'ordre est celui des passage_id, y compris au rang k."""
    ids = ["doc_c", "doc_a", "doc_d", "doc_b"]
    index = IndexVectoriel.construire(ids, _vecteurs([1, 0], [1, 0], [0, 1], [1, 0]))
    requete = _vecteurs([1, 0])[0]
    assert [pid for pid, _ in index.rechercher(requete, k=3)] == ["doc_a", "doc_b", "doc_c"]
    assert [pid for pid, _ in index.rechercher(requete, k=2)] == ["doc_a", "doc_b"]


def test_k_plus_grand_que_l_index_et_index_vide() -> None:
    index = IndexVectoriel.construire(["a"], _vecteurs([1, 0]))
    assert len(index.rechercher(_vecteurs([1, 0])[0], k=10)) == 1
    vide = IndexVectoriel.construire([], np.zeros((0, 2), dtype=np.float32))
    assert vide.rechercher(_vecteurs([1, 0])[0], k=5) == []


def test_ef09_index_recharge_donne_les_memes_resultats(tmp_path: Path) -> None:
    """EF09 : sauvegardé puis rechargé (dossier au nom accentué, cas Windows),
    l'index donne exactement les mêmes résultats."""
    rng = np.random.default_rng(0)
    vecteurs = rng.standard_normal((500, 32)).astype(np.float32)
    vecteurs /= np.linalg.norm(vecteurs, axis=1, keepdims=True)
    ids = [f"p{i:04d}" for i in range(500)]
    index = IndexVectoriel.construire(ids, vecteurs)
    dossier = tmp_path / "données_é"
    index.sauvegarder(dossier)

    recharge = IndexVectoriel.charger(dossier)
    for q in vecteurs[:20]:
        assert recharge.rechercher(q, 10) == index.rechercher(q, 10)


def test_ef09_rechargement_rapide_a_pleine_capacite(tmp_path: Path) -> None:
    """EF09 et ENF03 : 5 000 passages en 1 024 dimensions (bge-m3) rechargés en moins de 30 s."""
    vecteurs = np.random.default_rng(1).standard_normal((5_000, 1024)).astype(np.float32)
    IndexVectoriel.construire([f"p{i}" for i in range(5_000)], vecteurs).sauvegarder(tmp_path)
    debut = time.perf_counter()
    index = IndexVectoriel.charger(tmp_path)
    assert len(index) == 5_000
    assert time.perf_counter() - debut < 30


def test_index_incoherent_refuse() -> None:
    with pytest.raises(ValueError):
        IndexVectoriel.construire(["a", "b"], _vecteurs([1, 0]))


# --- Indexation : construction, rechargement, empreinte (D12) ---------------------------------


def test_construit_puis_recharge_sans_charger_le_modele(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    premier = indexer(PASSAGES, cfg, REVISION, lambda: EncodeurFactice(revision=REVISION))
    assert premier.statut == "construit"
    assert premier.raison == "aucun index pour cette configuration"

    second = indexer(PASSAGES, cfg, REVISION, _jamais_appelee)
    assert second.statut == "recharge"
    assert second.meta["empreinte"] == premier.meta["empreinte"]


def test_textes_encodes_exactement(tmp_path: Path) -> None:
    """Préfixe « passage: » + chemin de section + contenu : c'est ce texte qui est encodé."""
    encodeur = EncodeurFactice(revision=REVISION)
    indexer(PASSAGES, _config(tmp_path), REVISION, lambda: encodeur)
    assert encodeur.appels[0][0] == "passage: Pods\n\n" + PASSAGES[0].texte


MODIFICATIONS = {
    "prefixe_passage": lambda c: c.model_copy(
        update={"embeddings": c.embeddings.model_copy(update={"prefixe_passage": "doc: "})}
    ),
    "prefixe_requete": lambda c: c.model_copy(
        update={"embeddings": c.embeddings.model_copy(update={"prefixe_requete": "q: "})}
    ),
    "normaliser": lambda c: c.model_copy(
        update={"embeddings": c.embeddings.model_copy(update={"normaliser": False})}
    ),
    "modele": lambda c: c.model_copy(
        update={"embeddings": c.embeddings.model_copy(update={"modele": "autre/modele"})}
    ),
    "contexte_section": lambda c: c.model_copy(
        update={"decoupage": DecoupageConfig(contexte_section=False)}
    ),
}


@pytest.mark.parametrize("option", list(MODIFICATIONS))
def test_d12_changer_une_seule_option_change_l_empreinte(tmp_path: Path, option: str) -> None:
    base = _config(tmp_path)
    modifiee = MODIFICATIONS[option](base)

    def empreinte(c: Config) -> str:
        p = parametres_encodage(c.decoupage, c.embeddings, REVISION)
        return calculer_empreinte(PASSAGES, textes_encodes(PASSAGES, c.decoupage, c.embeddings), p)

    assert empreinte(modifiee) != empreinte(base)


@pytest.mark.parametrize("option", list(MODIFICATIONS))
def test_d12_changer_une_seule_option_force_la_reconstruction(tmp_path: Path, option: str) -> None:
    base = _config(tmp_path)
    indexer(PASSAGES, base, REVISION, lambda: EncodeurFactice(revision=REVISION))
    modifiee = MODIFICATIONS[option](base)
    encodeur = EncodeurFactice(
        revision=REVISION,
        normaliser=modifiee.embeddings.normaliser,
        modele=modifiee.embeddings.modele,
    )
    resultat = indexer(PASSAGES, modifiee, REVISION, lambda: encodeur)
    assert resultat.statut == "construit"
    assert encodeur.appels, "le modèle a bien été appelé pour reconstruire"


def test_d12_changer_la_revision_force_la_reconstruction(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = _config(tmp_path)
    indexer(PASSAGES, cfg, REVISION, lambda: EncodeurFactice(revision=REVISION))
    autre = "b" * 40
    with caplog.at_level(logging.WARNING):
        resultat = indexer(PASSAGES, cfg, autre, lambda: EncodeurFactice(revision=autre))
    assert resultat.statut == "construit"
    assert f"revision : '{REVISION}' → '{autre}'" in resultat.raison
    assert "Reconstruction de l'index" in caplog.text


@pytest.mark.parametrize(
    "passages",
    [
        [_passage(0, "Texte modifié."), *PASSAGES[1:]],
        [_passage(9, PASSAGES[0].texte, "Pods"), *PASSAGES[1:]],
        PASSAGES[:-1],
    ],
    ids=["texte", "identifiant", "passage_retire"],
)
def test_d12_passages_modifies_forcent_la_reconstruction(
    tmp_path: Path, passages: list[Passage]
) -> None:
    cfg = _config(tmp_path)
    indexer(PASSAGES, cfg, REVISION, lambda: EncodeurFactice(revision=REVISION))
    resultat = indexer(passages, cfg, REVISION, lambda: EncodeurFactice(revision=REVISION))
    assert resultat.statut == "construit"
    assert resultat.raison == "passages modifiés"


def test_deux_configurations_d_experience_coexistent(tmp_path: Path) -> None:
    base = _config(tmp_path)
    autre = MODIFICATIONS["prefixe_passage"](base)
    a = indexer(PASSAGES, base, REVISION, lambda: EncodeurFactice(revision=REVISION))
    b = indexer(PASSAGES, autre, REVISION, lambda: EncodeurFactice(revision=REVISION))
    assert a.dossier != b.dossier
    assert indexer(PASSAGES, base, REVISION, _jamais_appelee).statut == "recharge"
    assert indexer(PASSAGES, autre, REVISION, _jamais_appelee).statut == "recharge"


def test_d11_passages_tronques_comptes_et_signales(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """La limite porte sur le texte encodé (préfixe + chemin + contenu + tokens spéciaux)."""
    encodeur = EncodeurFactice(revision=REVISION, longueur_max=12)
    with caplog.at_level(logging.WARNING):
        resultat = indexer(PASSAGES, _config(tmp_path), REVISION, lambda: encodeur)
    # « passage: Pods » (2) + 12 mots + 2 tokens spéciaux = 16 > 12 ; le passage 3 tient (12)
    assert resultat.meta["passages_tronques"] == 3
    assert resultat.meta["exemples_tronques"] == ["doc_0000", "doc_0001", "doc_0002"]
    assert "seront tronqués" in caplog.text


def test_normalisation_verifiee(tmp_path: Path) -> None:
    encodeur_non_normalise = EncodeurFactice(revision=REVISION, normaliser=False)
    with pytest.raises(ValueError, match="norme de 1"):
        indexer(PASSAGES, _config(tmp_path), REVISION, lambda: encodeur_non_normalise)


def test_meta_de_l_index(tmp_path: Path) -> None:
    meta = indexer(
        PASSAGES, _config(tmp_path), REVISION, lambda: EncodeurFactice(revision=REVISION)
    ).meta
    assert meta["revision"] == REVISION
    assert meta["parametres"]["prefixe_passage"] == "passage: "
    assert meta["parametres"]["contexte_section"] is True
    assert meta["dimension"] == 256
    assert meta["passages"] == 4
    assert meta["duree_encodage_s"] >= 0
    assert set(meta["versions"]) == {
        "torch",
        "sentence-transformers",
        "transformers",
        "faiss-cpu",
        "numpy",
    }


def test_index_perime_refuse_pour_la_recherche(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    indexer(PASSAGES, cfg, REVISION, lambda: EncodeurFactice(revision=REVISION))
    charger_index_a_jour(PASSAGES, cfg, REVISION)  # à jour : pas d'erreur
    with pytest.raises(IndexPerime, match="rag ingest"):
        charger_index_a_jour(PASSAGES[:-1], cfg, REVISION)


# --- Recherche (EF11) -------------------------------------------------------------------------


def _moteur(tmp_path: Path) -> tuple[MoteurRecherche, EncodeurFactice]:
    cfg = _config(tmp_path)
    encodeur = EncodeurFactice(revision=REVISION)
    resultat = indexer(PASSAGES, cfg, REVISION, lambda: encodeur)
    return MoteurRecherche(resultat.index, PASSAGES, encodeur, cfg.embeddings), encodeur


def test_ef11_k_parametrable_et_score_par_passage(tmp_path: Path) -> None:
    moteur, _ = _moteur(tmp_path)
    for k in (1, 3, 4):
        resultats = moteur.rechercher("How do I create a Pod?", k)
        assert len(resultats) == k
        assert [r.rang for r in resultats] == list(range(1, k + 1))
        scores = [r.score for r in resultats]
        assert scores == sorted(scores, reverse=True)
        assert all(isinstance(s, float) for s in scores)


def test_ef11_passage_pertinent_en_tete(tmp_path: Path) -> None:
    moteur, _ = _moteur(tmp_path)
    # Encodeur factice : mots discriminants. La pertinence d'un vrai modèle est testée en « slow ».
    resultat = moteur.rechercher("Secret sensitive passwords", 1)[0]
    assert resultat.passage.passage_id == "doc_0002"


def test_prefixe_de_requete_applique(tmp_path: Path) -> None:
    moteur, encodeur = _moteur(tmp_path)
    moteur.rechercher("Qu'est-ce qu'un Pod ?", 2)
    assert encodeur.appels[-1] == ["query: Qu'est-ce qu'un Pod ?"]


def test_index_et_passages_incoherents_refuses(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    encodeur = EncodeurFactice(revision=REVISION)
    resultat = indexer(PASSAGES, cfg, REVISION, lambda: encodeur)
    with pytest.raises(ValueError, match="introuvables"):
        MoteurRecherche(resultat.index, PASSAGES[:2], encodeur, cfg.embeddings)


# --- Révision du modèle (D4) ------------------------------------------------------------------


def test_revision_figee_renvoyee_sans_reseau(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    def interdit(*args: object, **kwargs: object) -> str:
        raise AssertionError("aucun accès réseau pour une révision figée")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", interdit)
    assert resoudre_revision("intfloat/multilingual-e5-small", REVISION) == REVISION


def test_revision_main_resolue_avec_avertissement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import huggingface_hub

    instantane = tmp_path / "snapshots" / ("c" * 40)
    instantane.mkdir(parents=True)
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda *a, **k: str(instantane / "config.json")
    )
    with caplog.at_level(logging.WARNING):
        assert resoudre_revision("intfloat/multilingual-e5-small", "main") == "c" * 40
    assert "non figée" in caplog.text and "c" * 40 in caplog.text


def test_revision_inaccessible(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    def hors_ligne(*args: object, **kwargs: object) -> str:
        raise OSError("réseau coupé")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hors_ligne)
    with pytest.raises(ModeleIndisponible, match=r"huggingface\.co"):
        resoudre_revision("intfloat/multilingual-e5-small", "main")
