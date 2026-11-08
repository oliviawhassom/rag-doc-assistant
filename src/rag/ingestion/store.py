"""Stockage des documents importés et de leurs pages (EF04).

Organisation sur disque (dossier ``stockage.dossier`` de la configuration) :

    store/
      documents.jsonl          un Document par ligne : le registre du corpus
      pages/<document_id>.jsonl  une Page par ligne, dans l'ordre

Pourquoi des fichiers JSONL plutôt qu'une base SQLite : le volume est petit
(≤ 1 000 pages), le format se lit avec n'importe quel éditeur et se compare
facilement entre deux exécutions. Le registre est entièrement chargé en
mémoire au démarrage.

Ordre d'écriture : les pages d'abord, puis la ligne du registre. Si le
programme s'arrête entre les deux, le registre ne pointe jamais vers des pages
manquantes ; au pire un fichier de pages orphelin est réécrit au prochain import.
"""

from __future__ import annotations

import os
from pathlib import Path

from rag.models import Document, Page


def document_id_from_hash(empreinte: str) -> str:
    """Identifiant déterministe dérivé du contenu.

    Un même fichier reçoit toujours le même identifiant, quelle que soit la
    machine ou l'ordre d'import : les citations et le jeu de test (qui
    référence des ``document_id``) restent valides après une réindexation
    (ENF06). 12 caractères hexadécimaux = 48 bits : aucune collision attendue
    à l'échelle de quelques milliers de documents.
    """
    return f"doc_{empreinte[:12]}"


class DocumentStore:
    def __init__(self, dossier: Path) -> None:
        self.dossier = dossier
        self.chemin_registre = dossier / "documents.jsonl"
        self.dossier_pages = dossier / "pages"
        self._par_empreinte: dict[str, Document] = {}
        self._par_id: dict[str, Document] = {}
        self._charger()

    def _charger(self) -> None:
        if not self.chemin_registre.exists():
            return
        with self.chemin_registre.open(encoding="utf-8") as f:
            for ligne in f:
                if ligne.strip():
                    self._indexer(Document.model_validate_json(ligne))

    def _indexer(self, doc: Document) -> None:
        self._par_empreinte[doc.empreinte] = doc
        self._par_id[doc.document_id] = doc

    def find_by_hash(self, empreinte: str) -> Document | None:
        return self._par_empreinte.get(empreinte)

    def get(self, document_id: str) -> Document | None:
        return self._par_id.get(document_id)

    def list_documents(self) -> list[Document]:
        return sorted(self._par_id.values(), key=lambda d: d.nom_fichier)

    def add(self, doc: Document, pages: list[Page]) -> None:
        if doc.empreinte in self._par_empreinte:
            raise ValueError(f"Document déjà présent : {doc.document_id}")
        if len(pages) != doc.nb_pages:
            raise ValueError("Le nombre de pages ne correspond pas à nb_pages")

        self.dossier_pages.mkdir(parents=True, exist_ok=True)
        cible = self.dossier_pages / f"{doc.document_id}.jsonl"
        temporaire = cible.with_suffix(".jsonl.tmp")
        with temporaire.open("w", encoding="utf-8") as f:
            for page in pages:
                f.write(page.model_dump_json() + "\n")
        os.replace(temporaire, cible)  # renommage atomique

        with self.chemin_registre.open("a", encoding="utf-8") as f:
            f.write(doc.model_dump_json() + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._indexer(doc)

    def load_pages(self, document_id: str) -> list[Page]:
        chemin = self.dossier_pages / f"{document_id}.jsonl"
        with chemin.open(encoding="utf-8") as f:
            return [Page.model_validate_json(ligne) for ligne in f if ligne.strip()]
