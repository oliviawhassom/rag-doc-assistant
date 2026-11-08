"""Erreurs d'ingestion (EF03).

Chaque cause de rejet a sa propre classe, avec :
- ``code`` : identifiant stable, utilisable dans les journaux et les tests ;
- ``http_status`` : le code que renverra l'API REST en V2 (section 6.2), défini
  ici pour que la correspondance cause → code HTTP n'existe qu'à un seul endroit ;
- un message en français, lisible par l'utilisateur, qui indique la cause.
"""

from __future__ import annotations


class IngestionError(Exception):
    code = "erreur_ingestion"
    http_status = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UnsupportedFormat(IngestionError):
    code = "format_non_accepte"
    http_status = 415


class FileTooLarge(IngestionError):
    code = "fichier_trop_lourd"
    http_status = 413


class EmptyFile(IngestionError):
    code = "fichier_vide"
    http_status = 422


class InvalidEncoding(IngestionError):
    code = "encodage_invalide"
    http_status = 422


class UnreadablePdf(IngestionError):
    code = "pdf_illisible"
    http_status = 422


class PasswordProtectedPdf(IngestionError):
    """PDF qui exige un mot de passe d'ouverture. Un PDF chiffré qui s'ouvre sans
    mot de passe (simples restrictions d'usage) n'est PAS rejeté : voir extractors."""

    code = "pdf_mot_de_passe_requis"
    http_status = 422


class PdfWithoutText(IngestionError):
    code = "pdf_sans_texte"
    http_status = 422
