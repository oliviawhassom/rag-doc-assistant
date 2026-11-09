"""Tests avec le vrai modèle multilingual-e5-small (décision D8).

L'encodeur factice des autres tests ne vérifie justement pas ce qui justifie D8 :
le calcul du vecteur par le modèle (moyenne des tokens, normalisation). Ces
tests le vérifient sur des cas évidents.

Marqués « slow » : exclus par défaut. Les lancer avec :

    python -m pytest -m slow

Premier lancement : téléchargement du modèle (environ 470 Mo, mis en cache).
Sautés si le modèle est inaccessible (pas de réseau ni de cache).
"""

import numpy as np
import pytest

from rag.config import EmbeddingsConfig
from rag.index import Encodeur, ModeleIndisponible, charger_encodeur, resoudre_revision

pytestmark = pytest.mark.slow

CFG = EmbeddingsConfig()  # multilingual-e5-small, préfixes e5, normalisation

PASSAGES = {
    "pod": "A Pod is the smallest deployable unit of computing in Kubernetes. You create a Pod "
    "by writing a manifest that lists its containers and applying it with kubectl apply.",
    "service": "A Service exposes a network application that is running as one or more Pods. "
    "It gives a stable IP address and DNS name and load-balances traffic across the Pods.",
    "secret": "A Secret is an object that contains a small amount of sensitive data such as a "
    "password, a token, or a key, so that you don't need to include it in application code.",
    "volume": "A PersistentVolume is a piece of storage in the cluster, provisioned by an "
    "administrator or dynamically using a StorageClass, with a lifecycle independent of Pods.",
}

# (question, passage attendu, passage qui doit être moins proche)
PAIRES = [
    ("How do I create a Pod?", "pod", "service"),
    ("How can I give my application a stable IP address and DNS name?", "service", "pod"),
    ("Where should I store a database password?", "secret", "volume"),
    # D2 : question en français, passages en anglais (modèle multilingue)
    ("Comment rendre le stockage indépendant du cycle de vie des Pods ?", "volume", "secret"),
]


@pytest.fixture(scope="module")
def e5() -> Encodeur:
    try:
        return charger_encodeur(CFG, resoudre_revision(CFG.modele, CFG.revision))
    except ModeleIndisponible as exc:
        pytest.skip(f"modèle réel indisponible : {exc}")


@pytest.fixture(scope="module")
def vecteurs_passages(e5: Encodeur) -> dict[str, np.ndarray]:
    noms = list(PASSAGES)
    vecteurs = e5.encoder([CFG.prefixe_passage + PASSAGES[n] for n in noms])
    return dict(zip(noms, vecteurs, strict=True))


def test_dimension_et_revision(e5: Encodeur) -> None:
    assert e5.dimension == 384
    assert e5.longueur_max == 512
    assert len(e5.revision) == 40


def test_vecteurs_de_norme_1(e5: Encodeur, vecteurs_passages: dict[str, np.ndarray]) -> None:
    normes = np.linalg.norm(np.stack(list(vecteurs_passages.values())), axis=1)
    assert np.allclose(normes, 1.0, atol=1e-4)
    requete = e5.encoder([CFG.prefixe_requete + "test"])[0]
    assert abs(float(np.linalg.norm(requete)) - 1.0) < 1e-4


@pytest.mark.parametrize(("question", "attendu", "autre"), PAIRES)
def test_question_plus_proche_du_bon_passage(
    e5: Encodeur,
    vecteurs_passages: dict[str, np.ndarray],
    question: str,
    attendu: str,
    autre: str,
) -> None:
    q = e5.encoder([CFG.prefixe_requete + question])[0]
    bon = float(q @ vecteurs_passages[attendu])
    mauvais = float(q @ vecteurs_passages[autre])
    assert bon > mauvais, f"{question!r} : {attendu}={bon:.4f} ≤ {autre}={mauvais:.4f}"


QUESTIONS_ANGLAISES = [p for p in PAIRES if p[0].isascii()]


def test_meilleur_passage_parmi_les_quatre_questions_anglaises(
    e5: Encodeur, vecteurs_passages: dict[str, np.ndarray]
) -> None:
    """Pour les questions anglaises, le bon passage est premier parmi les quatre.

    La question française en est exclue, à dessein. Observé le 9 novembre 2026
    avec e5-small : « Comment rendre le stockage indépendant du cycle de vie des
    Pods ? » classe le passage sur les Pods premier (0,824), devant celui sur les
    volumes (0,811) ; le mot « Pods », présent dans la question, l'emporte sur le
    sens de la phrase française. Le test par paires (volume > secret) passe.
    C'est une limite de qualité du modèle en recherche multilingue, pas un défaut
    de calcul : elle se mesure sur le jeu de test (Recall@5 par langue, D2), pas
    dans un test unitaire.
    """
    for question, attendu, _ in QUESTIONS_ANGLAISES:
        q = e5.encoder([CFG.prefixe_requete + question])[0]
        scores = {n: float(q @ v) for n, v in vecteurs_passages.items()}
        assert max(scores, key=scores.__getitem__) == attendu, (question, scores)


def test_pas_de_prefixe_ajoute_par_sentence_transformers(e5: Encodeur) -> None:
    """Nos préfixes ne doivent pas être doublés par un préfixe par défaut du modèle :
    encoder « query: x » doit différer d'encoder « query: query: x »."""
    simple, double = e5.encoder(["query: What is a Pod?", "query: query: What is a Pod?"])
    assert float(simple @ double) < 0.9999


def test_comptage_des_tokens_avec_tokens_speciaux(e5: Encodeur) -> None:
    """D11 : la longueur encodée compte les tokens spéciaux (<s> et </s> pour XLM-RoBERTa)."""
    assert e5.compter_tokens("passage: Pod") >= 4
