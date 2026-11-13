import pytest

from realestate_agent.rag import KnowledgeBase, chunk_markdown


@pytest.fixture(scope="module")
def kb(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("RE_AGENT_RUNTIME", str(tmp_path_factory.mktemp("rag")))
        yield KnowledgeBase()


def test_ingests_at_least_five_documents(kb):
    stats = kb.ingest()
    assert stats["documents"] >= 5
    assert stats["chunks"] > stats["documents"]


@pytest.mark.parametrize(
    "query,expected",
    [
        ("How much deposit should I pay?", "01_buying_process.md"),
        ("What should a lawyer check about ownership and the title deed?", "02_title_verification.md"),
        ("What share of my income can go to the monthly loan payment?", "03_financing_affordability.md"),
        ("How do I check for damp and cracks when inspecting a house?", "04_inspection_guide.md"),
        ("Does Allama Iqbal Town have waterlogging in the monsoon?", "05_neighbourhood_guide.md"),
        ("Is a viewing request a guaranteed appointment?", "06_advisory_policy.md"),
        ("Who pays stamp duty and advance tax at transfer?", "07_taxes_and_transfer_costs.md"),
        ("Is the housing society approved by the LDA?", "08_housing_societies_and_approvals.md"),
        ("How many marla are in one kanal?", "09_measurements_marla_kanal.md"),
        ("What is a fard and an intiqal?", "10_documents_glossary.md"),
    ],
)
def test_top_result_comes_from_expected_document(kb, query, expected):
    results = kb.retrieve(query)
    assert results, "expected at least one passage"
    assert results[0]["source"] == expected
    assert results[0]["citation"].startswith(expected + "#")


@pytest.mark.parametrize(
    "query", ["What is the boiling point of tungsten?", "Best chocolate cake recipe with frosting"]
)
def test_unrelated_query_returns_nothing(kb, query):
    assert kb.retrieve(query) == []


def test_chunks_respect_size_limit_and_keep_all_sections():
    text = "# Doc\n\n## One\n\n" + "a " * 1000 + "\n\n## Two\n\nshort"
    chunks = chunk_markdown(text)
    assert {title for title, _ in chunks} >= {"One", "Two"}
    assert all(len(body) <= 800 + len("One: ") for _, body in chunks)
    assert len([c for c in chunks if c[0] == "One"]) > 1


LEXICAL_CASES = [
    ("How much deposit should I pay?", "01_buying_process.md"),
    ("What should a lawyer check about ownership and the title deed?", "02_title_verification.md"),
    ("What share of my income can go to the monthly loan payment?", "03_financing_affordability.md"),
    ("How do I check for damp and cracks when inspecting a house?", "04_inspection_guide.md"),
    ("Does Allama Iqbal Town have waterlogging in the monsoon?", "05_neighbourhood_guide.md"),
    ("Is a viewing request a guaranteed appointment?", "06_advisory_policy.md"),
]


@pytest.mark.parametrize("query,expected", LEXICAL_CASES)
def test_lexical_retriever_finds_expected_document(query, expected):
    from realestate_agent.rag import LexicalKnowledgeBase

    results = LexicalKnowledgeBase().retrieve(query)
    assert results and results[0]["source"] == expected
    assert results[0]["citation"].startswith(expected + "#")


@pytest.mark.parametrize(
    "query", ["What is the boiling point of tungsten?", "Best chocolate cake recipe with frosting"]
)
def test_lexical_retriever_ignores_unrelated_queries(query):
    from realestate_agent.rag import LexicalKnowledgeBase

    assert LexicalKnowledgeBase().retrieve(query) == []


def test_hybrid_rejects_off_topic_even_with_shared_words(tmp_path_factory):
    from realestate_agent.rag import HybridKnowledgeBase

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("RE_AGENT_RUNTIME", str(tmp_path_factory.mktemp("hybrid")))
        hybrid = HybridKnowledgeBase()
    assert hybrid.retrieve("What is the capital of France?") == []
    hits = hybrid.retrieve("How many square feet is a kanal?")
    assert hits[0]["source"] == "09_measurements_marla_kanal.md"
    assert set(hits[0]["matched_by"]) == {"dense", "lexical"}
    assert hybrid.passage(hits[0]["citation"])["text"] == hits[0]["text"]
    assert hybrid.passage("nope.md#0") is None
