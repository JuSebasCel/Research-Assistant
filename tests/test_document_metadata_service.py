"""Tests de DocumentMetadataService: round-trip JSON, semántica de overwrite."""

import pytest

from rag_app.services.document_metadata_service import DocumentMetadataService


@pytest.fixture
def service(tmp_path) -> DocumentMetadataService:
    return DocumentMetadataService(tmp_path / "document_metadata.json")


def test_get_unknown_document_returns_empty_dict(service):
    assert service.get("no_existe") == {}


def test_set_display_name_persists_and_round_trips(service):
    service.set_display_name("doc_a", "Título A")

    assert service.get("doc_a") == {"display_name": "Título A"}


def test_set_display_name_overwrite_false_does_not_clobber_existing(service):
    service.set_display_name("doc_a", "Nombre original")
    service.set_display_name("doc_a", "Nombre auto-extraído", overwrite=False)

    assert service.get("doc_a")["display_name"] == "Nombre original"


def test_set_display_name_overwrite_false_sets_when_absent(service):
    service.set_display_name("doc_a", "Primer nombre", overwrite=False)

    assert service.get("doc_a")["display_name"] == "Primer nombre"


def test_set_display_name_overwrite_true_replaces_existing(service):
    service.set_display_name("doc_a", "Nombre viejo")
    service.set_display_name("doc_a", "Nombre nuevo", overwrite=True)

    assert service.get("doc_a")["display_name"] == "Nombre nuevo"


def test_set_folder_persists_and_can_be_cleared(service):
    service.set_folder("doc_a", "Exosomas")
    assert service.get("doc_a")["folder"] == "Exosomas"

    service.set_folder("doc_a", None)
    assert service.get("doc_a")["folder"] is None


def test_get_all_returns_every_document(service):
    service.set_display_name("doc_a", "A")
    service.set_folder("doc_b", "Carpeta")

    all_metadata = service.get_all()

    assert set(all_metadata.keys()) == {"doc_a", "doc_b"}


def test_new_instance_reads_persisted_state_from_disk(tmp_path):
    path = tmp_path / "document_metadata.json"
    DocumentMetadataService(path).set_display_name("doc_a", "Persistido")

    reloaded = DocumentMetadataService(path)

    assert reloaded.get("doc_a")["display_name"] == "Persistido"
