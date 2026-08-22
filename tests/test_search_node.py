from unittest.mock import patch

from src.models import SourceDocument
from src.nodes.search_node import search_node

DOC = SourceDocument(
    url="https://reuters.com/a", title="T", snippet="s", content="c"
)


def test_returns_documents_for_the_claim():
    with patch("src.nodes.search_node.search_evidence", return_value=[DOC]) as mock:
        result = search_node({"claim": "did X happen"})

    mock.assert_called_once_with("did X happen")
    assert result == {"documents": [DOC]}


def test_no_results_yields_empty_document_list():
    with patch("src.nodes.search_node.search_evidence", return_value=[]):
        assert search_node({"claim": "obscure"}) == {"documents": []}
