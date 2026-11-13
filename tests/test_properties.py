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


def test_generated_listings_cover_more_areas_with_coordinates():
    homes = properties.search(include_unavailable=True)
    assert len(homes) >= 60
    assert {"DHA Phase 6", "Bahria Town", "Wapda Town"} <= {h["area"] for h in homes}
    assert all(h["lat"] and h["lng"] and h["marla"] > 0 for h in homes)


def test_area_price_stats():
    stats = {s["area"]: s for s in properties.area_price_stats()}
    assert len(stats) == 14
    gulberg = stats["Gulberg"]
    assert gulberg["min_price_per_sqft"] <= gulberg["median_price_per_sqft"] <= gulberg["max_price_per_sqft"]
    # Per-sqft median is rounded before display, so allow up to one rupee per sqft of drift.
    assert abs(gulberg["median_price_per_marla"] - gulberg["median_price_per_sqft"] * 225) <= 225
    with pytest.raises(ValueError):
        properties.area_price_stats("Atlantis")


def test_affordability_uses_35_percent_rule():
    out = properties.affordability(300_000, 5_000_000)
    assert out["monthly_payment_pkr"] == 105_000
    # The loan that 105,000 a month repays matches the payment calculator.
    loan_check = properties.monthly_payment(out["max_loan_pkr"] + 1, 0, 14, 20)
    assert abs(loan_check["monthly_payment_pkr"] - 105_000) < 200
    assert properties.affordability(100_000, 1_000_000, existing_debt_pkr=40_000)["max_loan_pkr"] == 0


def test_plot_size_filters_cover_kanal_houses_and_farmhouses():
    kanal = properties.search(min_marla=19.5, max_marla=20.5, include_unavailable=True)
    assert kanal and all(19.5 <= h["marla"] <= 20.5 for h in kanal)
    big = properties.search(min_marla=100.5, include_unavailable=True)
    assert any(h["marla"] >= 200 for h in big)  # 10 kanal farmhouses exist.
    small = properties.search(max_marla=5.5)
    assert small and all(h["marla"] <= 5.5 for h in small)


def test_farmhouses_sit_on_the_farm_belts():
    farms = properties.search(property_type="farmhouse", include_unavailable=True)
    assert len(farms) >= 10
    assert {h["area"] for h in farms} <= {"Bedian Road", "Barki Road", "Raiwind Road"}
    assert all(h["marla"] >= 40 for h in farms)  # At least 2 kanal.


def test_generated_listings_have_pakistani_photos_with_attribution():
    import json
    from pathlib import Path
    photos = json.loads((Path(__file__).resolve().parent.parent / "realestate_agent" / "static" / "homes"
                         / "photos.json").read_text(encoding="utf-8"))
    homes = properties.search(include_unavailable=True)
    assert all(h["id"] in photos for h in homes)
    generated = [p for h in homes if int(h["id"][2:]) >= 113 for p in photos[h["id"]]]
    pk = [p for p in generated if p["src"].startswith("/static/homes/pk/")]
    assert pk and all(p["attribution"] and p["license_url"] for p in pk)
