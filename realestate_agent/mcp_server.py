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
) -> str:
    """Search current listings. Use for any question about which homes exist or match criteria.

    Args:
        max_price: Highest price in PKR.
        min_beds: Minimum number of bedrooms.
        area: Exact neighbourhood in Lahore: Johar Town, Allama Iqbal Town, Model Town, Park View City or Gulberg.
        property_type: house, townhouse or apartment.
        include_unavailable: Also return homes that are sold or under offer.
    """
    return json.dumps(
        properties.search(max_price, min_beds, area, property_type, include_unavailable)
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
