"""MCP server exposing the listing tools over stdio. Run: python -m realestate_agent.mcp_server"""
import json

from mcp.server.fastmcp import FastMCP

from . import properties

mcp = FastMCP("real-estate-listings")


@mcp.tool()
def search_properties(
    max_price: int | None = None,
    min_beds: int | None = None,
    area: str | None = None,
    property_type: str | None = None,
    include_unavailable: bool = False,
    min_marla: float | None = None,
    max_marla: float | None = None,
) -> str:
    """Search current listings. Use for any question about which homes exist or match criteria.

    Args:
        max_price: Highest price in PKR.
        min_beds: Minimum number of bedrooms.
        area: Exact neighbourhood in Lahore: Johar Town, Allama Iqbal Town, Model Town, Park View City,
            Gulberg, DHA Phase 6, DHA Phase 5, Lahore Cantt, Bahria Town, Lake City, Wapda Town,
            or the farmhouse belts Bedian Road, Barki Road and Raiwind Road.
        property_type: house, townhouse, apartment or farmhouse.
        include_unavailable: Also return homes that are sold or under offer.
        min_marla: Smallest plot size in marla. 1 kanal = 20 marla, so "1 kanal or more" is 19.5.
        max_marla: Largest plot size in marla. "Up to 5 marla" is 5.5.
    """
    return json.dumps(
        properties.search(max_price, min_beds, area, property_type, include_unavailable,
                          min_marla, max_marla)
    )


@mcp.tool()
def get_property(property_id: str) -> str:
    """Get live details and status of one listing. Use to re-check a home before recommending it.

    Args:
        property_id: Listing id, for example P-103.
    """
    return json.dumps(properties.get(property_id))


@mcp.tool()
def compare_properties(
    property_ids: list[str],
    budget: int | None = None,
    min_beds: int | None = None,
    preferred_areas: list[str] | None = None,
    must_have_features: list[str] | None = None,
    max_commute_min: int | None = None,
) -> str:
    """Compare two or more listings against the buyer's criteria and rank the available ones.

    Args:
        property_ids: Listing ids to compare.
        budget: Buyer's maximum price in PKR.
        min_beds: Minimum bedrooms.
        preferred_areas: Neighbourhoods the buyer prefers.
        must_have_features: Required features, for example garden, parking, school_nearby.
        max_commute_min: Longest acceptable commute to the centre, in minutes.
    """
    return json.dumps(
        properties.compare(
            property_ids, budget, min_beds, preferred_areas, must_have_features, max_commute_min
        )
    )


@mcp.tool()
def estimate_monthly_payment(
    price: int, down_payment_pct: float, annual_rate_pct: float, years: int
) -> str:
    """Calculate an illustrative monthly loan payment. Never estimate this by hand.

    Args:
        price: Home price in PKR.
        down_payment_pct: Down payment as a percent of price.
        annual_rate_pct: Annual interest rate as a percent.
        years: Loan term in years.
    """
    return json.dumps(properties.monthly_payment(price, down_payment_pct, annual_rate_pct, years))


@mcp.tool()
def area_price_stats(area: str | None = None) -> str:
    """Median asking price, price per square foot and price per marla by Lahore area.

    Use for "what do homes cost in X" or to judge whether a listing is priced high or low for
    its area. Figures come from the synthetic listings, not the real market.

    Args:
        area: One area, or leave empty for every area.
    """
    return json.dumps(properties.area_price_stats(area))


@mcp.tool()
def affordability_check(
    monthly_income_pkr: int,
    down_payment_pkr: int,
    existing_debt_pkr: int = 0,
    annual_rate_pct: float = 14.0,
    years: int = 20,
) -> str:
    """Highest home price a buyer can plan for, using the financing guide's 35 percent rule.

    Args:
        monthly_income_pkr: Steady household income per month, in PKR.
        down_payment_pkr: Cash available for the down payment, in PKR.
        existing_debt_pkr: Monthly payments on other loans, in PKR.
        annual_rate_pct: Annual interest rate as a percent. The guide's planning rate is 14.
        years: Loan term in years. The guide's planning term is 20.
    """
    return json.dumps(properties.affordability(
        monthly_income_pkr, down_payment_pkr, annual_rate_pct, years, 35.0, existing_debt_pkr))


@mcp.tool()
def request_viewing(property_id: str, buyer_name: str, preferred_date: str) -> str:
    """Send a simulated viewing request to the listing office. The buyer must approve it first.

    Args:
        property_id: Listing id.
        buyer_name: Name for the request.
        preferred_date: Buyer's preferred date as YYYY-MM-DD.
    """
    return json.dumps(properties.request_viewing(property_id, buyer_name, preferred_date))


if __name__ == "__main__":
    mcp.run(transport="stdio")
