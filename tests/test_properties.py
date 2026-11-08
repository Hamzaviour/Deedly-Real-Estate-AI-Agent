import pytest

from realestate_agent import properties


def test_search_applies_filters_and_hides_unavailable():
    homes = properties.search(max_price=27_000_000, min_beds=3)
    ids = [h["id"] for h in homes]
    assert "P-103" in ids and "P-110" in ids
    assert all(h["price_pkr"] <= 27_000_000 and h["beds"] >= 3 for h in homes)
    assert all(h["status"] == "available" for h in homes)


def test_search_can_include_unavailable():
    ids = [h["id"] for h in properties.search(area="Allama Iqbal Town", include_unavailable=True)]
    assert "P-112" in ids


def test_get_unknown_raises():
    with pytest.raises(ValueError):
        properties.get("P-999")


def test_compare_ranks_by_criteria_met():
    out = properties.compare(
        ["P-103", "P-111"], budget=27_000_000, min_beds=3, must_have_features=["parking"]
    )
    assert [h["id"] for h in out["ranked"]] == ["P-103", "P-111"]
    assert out["ranked"][0]["criteria_met"] == 3
    assert out["ranked"][1]["checks"]["has_parking"] is False


def test_compare_breaks_ties_on_price_per_sqft():
    out = properties.compare(["P-103", "P-110"], budget=27_000_000, min_beds=3)
    assert out["ranked"][0]["criteria_met"] == out["ranked"][1]["criteria_met"]
    assert out["ranked"][0]["price_per_sqft"] <= out["ranked"][1]["price_per_sqft"]


def test_compare_excludes_unavailable_and_reports_missing():
    out = properties.compare(["P-103", "P-112", "P-999"], budget=60_000_000)
    assert [h["id"] for h in out["ranked"]] == ["P-103"]
    assert out["unavailable"] == [{"id": "P-112", "status": "sold"}]
    assert out["not_found"] == ["P-999"]


def test_compare_needs_two_homes():
    with pytest.raises(ValueError):
        properties.compare(["P-103", "P-103"])


def test_monthly_payment_known_value():
    # 1,000,000 at 12% over 1 year: standard formula gives 88,849 per month.
    out = properties.monthly_payment(1_000_000, 0, 12, 1)
    assert out["monthly_payment_pkr"] == 88849


def test_monthly_payment_zero_rate():
    assert properties.monthly_payment(1_200_000, 0, 0, 1)["monthly_payment_pkr"] == 100_000


@pytest.mark.parametrize(
    "args", [(0, 25, 14, 20), (100, 100, 14, 20), (100, 25, -1, 20), (100, 25, 14, 0)]
)
def test_monthly_payment_rejects_bad_input(args):
    with pytest.raises(ValueError):
        properties.monthly_payment(*args)


def test_viewing_blocked_after_change_event():
    assert properties.request_viewing("P-103", "Hamza", "2026-11-20")["simulated"] is True
    properties.set_status("P-103", "sold")
    with pytest.raises(ValueError, match="sold"):
        properties.request_viewing("P-103", "Hamza", "2026-11-20")
