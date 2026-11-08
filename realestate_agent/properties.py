"""Synthetic listing store and deterministic comparison logic behind the MCP server."""
import json
import sqlite3
from pathlib import Path

from .config import DATA_FILE, runtime_dir


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or runtime_dir() / "properties.db")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS properties (
            id TEXT PRIMARY KEY, title TEXT, area TEXT, type TEXT, price_pkr INTEGER,
            beds INTEGER, baths INTEGER, area_sqft INTEGER, year_built INTEGER,
            status TEXT, features TEXT, commute_min_to_center INTEGER, description TEXT)"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS viewing_requests (
            ref INTEGER PRIMARY KEY AUTOINCREMENT, property_id TEXT, buyer_name TEXT,
            preferred_date TEXT, status TEXT)"""
    )
    if conn.execute("SELECT COUNT(*) FROM properties").fetchone()[0] == 0:
        for p in json.loads(DATA_FILE.read_text(encoding="utf-8")):
            conn.execute(
                """INSERT INTO properties VALUES (:id, :title, :area, :type, :price_pkr, :beds,
                :baths, :area_sqft, :year_built, :status, :features,
                :commute_min_to_center, :description)""",
                {**p, "features": json.dumps(p["features"])},
            )
        conn.commit()
    return conn


def _row(row: sqlite3.Row) -> dict:
    data = dict(row)
    data["features"] = json.loads(data["features"])
    data["price_per_sqft"] = round(data["price_pkr"] / data["area_sqft"])
    return data


def search(
    max_price: int | None = None,
    min_beds: int | None = None,
    area: str | None = None,
    property_type: str | None = None,
    include_unavailable: bool = False,
) -> list[dict]:
    """Return listings matching every given filter, cheapest first."""
    sql, args = "SELECT * FROM properties WHERE 1=1", []
    if max_price is not None:
        sql += " AND price_pkr <= ?"
        args.append(max_price)
    if min_beds is not None:
        sql += " AND beds >= ?"
        args.append(min_beds)
    if area:
        sql += " AND LOWER(area) = LOWER(?)"
        args.append(area)
    if property_type:
        sql += " AND LOWER(type) = LOWER(?)"
        args.append(property_type)
    if not include_unavailable:
        sql += " AND status = 'available'"
    with _connect() as conn:
        return [_row(r) for r in conn.execute(sql + " ORDER BY price_pkr", args)]


def get(property_id: str) -> dict:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM properties WHERE id = ?", (property_id,)).fetchone()
    if row is None:
        raise ValueError(f"No listing with id {property_id}")
    return _row(row)


def compare(
    property_ids: list[str],
    budget: int | None = None,
    min_beds: int | None = None,
    preferred_areas: list[str] | None = None,
    must_have_features: list[str] | None = None,
    max_commute_min: int | None = None,
) -> dict:
    """Check each home against the buyer's criteria and rank the available ones.

    Ranking: most criteria met, then lowest price per square foot as the tie-breaker.
    Homes that are not available are reported but never ranked.
    """
    if len(set(property_ids)) < 2:
        raise ValueError("Comparison needs at least two different property ids")

    ranked, unavailable, not_found = [], [], []
    for pid in dict.fromkeys(property_ids):
        try:
            home = get(pid)
        except ValueError:
            not_found.append(pid)
            continue
        if home["status"] != "available":
            unavailable.append({"id": pid, "status": home["status"]})
            continue

        checks = {}
        if budget is not None:
            checks["within_budget"] = home["price_pkr"] <= budget
        if min_beds is not None:
            checks["enough_beds"] = home["beds"] >= min_beds
        if preferred_areas:
            checks["preferred_area"] = home["area"].lower() in {a.lower() for a in preferred_areas}
        if max_commute_min is not None:
            checks["commute_ok"] = home["commute_min_to_center"] <= max_commute_min
        for feature in must_have_features or []:
            checks[f"has_{feature}"] = feature in home["features"]

        ranked.append(
            {
                "id": pid,
                "title": home["title"],
                "area": home["area"],
                "price_pkr": home["price_pkr"],
                "beds": home["beds"],
                "area_sqft": home["area_sqft"],
                "year_built": home["year_built"],
                "commute_min_to_center": home["commute_min_to_center"],
                "price_per_sqft": home["price_per_sqft"],
                "checks": checks,
                "criteria_met": sum(checks.values()),
                "criteria_total": len(checks),
            }
        )

    ranked.sort(key=lambda h: (-h["criteria_met"], h["price_per_sqft"]))
    return {
        "ranking_rule": "most criteria met, then lowest price per sqft",
        "ranked": ranked,
        "unavailable": unavailable,
        "not_found": not_found,
    }


def monthly_payment(price: int, down_payment_pct: float, annual_rate_pct: float, years: int) -> dict:
    """Standard amortised loan payment. Illustrative only, not a lender quote."""
    if price <= 0 or years <= 0:
        raise ValueError("Price and years must be positive")
    if not 0 <= down_payment_pct < 100:
        raise ValueError("Down payment percent must be from 0 to below 100")
    if annual_rate_pct < 0:
        raise ValueError("Interest rate cannot be negative")

    loan = price * (1 - down_payment_pct / 100)
    months = years * 12
    rate = annual_rate_pct / 100 / 12
    payment = loan / months if rate == 0 else loan * rate / (1 - (1 + rate) ** -months)
    return {
        "loan_amount_pkr": round(loan),
        "monthly_payment_pkr": round(payment),
        "note": "Illustrative estimate. Confirm real terms with a lender.",
    }


def request_viewing(property_id: str, buyer_name: str, preferred_date: str) -> dict:
    """Record a simulated viewing request. Only available homes can be requested."""
    home = get(property_id)
    if home["status"] != "available":
        raise ValueError(f"{property_id} is {home['status']} and cannot be booked")
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO viewing_requests (property_id, buyer_name, preferred_date, status) "
            "VALUES (?, ?, ?, 'pending_office_confirmation')",
            (property_id, buyer_name, preferred_date),
        )
        return {
            "reference": f"VR-{cur.lastrowid:04d}",
            "status": "pending_office_confirmation",
            "simulated": True,
        }


def set_status(property_id: str, status: str) -> None:
    """Change event used by the demo and tests. Not exposed to the LLM."""
    get(property_id)
    with _connect() as conn:
        conn.execute("UPDATE properties SET status = ? WHERE id = ?", (status, property_id))
