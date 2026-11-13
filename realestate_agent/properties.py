"""Synthetic listing store and deterministic comparison logic behind the MCP server."""
import hashlib
import json
import sqlite3
import statistics
from pathlib import Path

from .config import DATA_FILE, runtime_dir


COLUMNS = ("id", "title", "area", "type", "price_pkr", "beds", "baths", "area_sqft", "year_built",
           "status", "features", "commute_min_to_center", "description", "marla", "lat", "lng")


def _data_version() -> int:
    """Changes whenever properties.json changes, so an outdated store is rebuilt."""
    return int(hashlib.sha1(DATA_FILE.read_bytes()).hexdigest()[:7], 16)


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or runtime_dir() / "properties.db")
    conn.row_factory = sqlite3.Row
    version = _data_version()
    if conn.execute("PRAGMA user_version").fetchone()[0] != version:
        conn.execute("DROP TABLE IF EXISTS properties")
        conn.execute(f"PRAGMA user_version = {version}")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS properties (
            id TEXT PRIMARY KEY, title TEXT, area TEXT, type TEXT, price_pkr INTEGER,
            beds INTEGER, baths INTEGER, area_sqft INTEGER, year_built INTEGER,
            status TEXT, features TEXT, commute_min_to_center INTEGER, description TEXT,
            marla REAL, lat REAL, lng REAL)"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS viewing_requests (
            ref INTEGER PRIMARY KEY AUTOINCREMENT, property_id TEXT, buyer_name TEXT,
            preferred_date TEXT, status TEXT)"""
    )
    if conn.execute("SELECT COUNT(*) FROM properties").fetchone()[0] == 0:
        for p in json.loads(DATA_FILE.read_text(encoding="utf-8")):
            row = {**p, "features": json.dumps(p["features"])}
            row.setdefault("marla", round(p["area_sqft"] / 225, 1))
            row.setdefault("lat", None)
            row.setdefault("lng", None)
            conn.execute(
                f"INSERT INTO properties VALUES ({', '.join(':' + c for c in COLUMNS)})", row
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
    min_marla: float | None = None,
    max_marla: float | None = None,
) -> list[dict]:
    """Return listings matching every given filter, cheapest first.

    Plot size filters are in marla (1 kanal = 20 marla), so a 1 kanal search is min_marla=19.5.
    """
    sql, args = "SELECT * FROM properties WHERE 1=1", []
    if min_marla is not None:
        sql += " AND marla >= ?"
        args.append(min_marla)
    if max_marla is not None:
        sql += " AND marla <= ?"
        args.append(max_marla)
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


def area_price_stats(area: str | None = None) -> list[dict]:
    """Median and range of asking price per square foot and per marla, by area.

    Built from every listing, sold ones included, because they still show the asking level.
    """
    with _connect() as conn:
        rows = [_row(r) for r in conn.execute("SELECT * FROM properties")]
    areas = sorted({r["area"] for r in rows})
    if area:
        match = [a for a in areas if a.lower() == area.strip().lower()]
        if not match:
            raise ValueError(f"Unknown area {area}. Known areas: {', '.join(areas)}")
        areas = match
    out = []
    for a in areas:
        homes = [r for r in rows if r["area"] == a]
        per_sqft = [r["price_per_sqft"] for r in homes]
        out.append({
            "area": a,
            "listings": len(homes),
            "available": sum(r["status"] == "available" for r in homes),
            "median_price_pkr": round(statistics.median(r["price_pkr"] for r in homes)),
            "median_price_per_sqft": round(statistics.median(per_sqft)),
            "median_price_per_marla": round(statistics.median(per_sqft) * 225),
            "min_price_per_sqft": min(per_sqft),
            "max_price_per_sqft": max(per_sqft),
            "median_commute_min": round(statistics.median(r["commute_min_to_center"] for r in homes)),
        })
    return out


def affordability(monthly_income_pkr: int, down_payment_pkr: int, annual_rate_pct: float = 14.0,
                  years: int = 20, max_payment_share_pct: float = 35.0,
                  existing_debt_pkr: int = 0) -> dict:
    """Highest home price whose loan payment stays within a share of monthly income.

    Defaults follow the financing guide: 14 percent, 20 years, and all loan payments, other debts
    included, at most 35 percent of steady income.
    """
    if monthly_income_pkr <= 0 or years <= 0:
        raise ValueError("Income and years must be positive")
    if down_payment_pkr < 0 or existing_debt_pkr < 0 or annual_rate_pct < 0:
        raise ValueError("Amounts and rate cannot be negative")
    if not 0 < max_payment_share_pct <= 60:
        raise ValueError("Payment share must be above 0 and at most 60 percent")
    budget = monthly_income_pkr * max_payment_share_pct / 100 - existing_debt_pkr
    if budget <= 0:
        return {"max_price_pkr": down_payment_pkr, "max_loan_pkr": 0, "monthly_payment_pkr": 0,
                "note": "Existing debt payments already use the whole payment allowance."}
    months, rate = years * 12, annual_rate_pct / 100 / 12
    loan = budget * months if rate == 0 else budget * (1 - (1 + rate) ** -months) / rate
    return {
        "max_price_pkr": round(loan + down_payment_pkr, -4),
        "max_loan_pkr": round(loan, -4),
        "monthly_payment_pkr": round(budget),
        "assumptions": {"annual_rate_pct": annual_rate_pct, "years": years,
                        "max_payment_share_pct": max_payment_share_pct},
        "note": "Illustrative ceiling, not a lender decision. Leave room for transfer costs.",
    }


def set_status(property_id: str, status: str) -> None:
    """Change event used by the demo and tests. Not exposed to the LLM."""
    get(property_id)
    with _connect() as conn:
        conn.execute("UPDATE properties SET status = ? WHERE id = ?", (status, property_id))
