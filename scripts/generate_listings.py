"""Generate the synthetic listing set in data/properties.json and the photo map in photos.json.

Three layers, all deterministic (fixed seeds):
1. The first 12 hand-written listings (P-101 to P-112) are kept as they are, because tests and
   docs refer to them.
2. 60 generated listings (P-113 to P-172) across the first 8 Lahore areas. Their random
   stream is unchanged from earlier versions, so their ids, prices and sizes stay stable.
3. A wider set (P-173 onward) that adds variety: kanal houses (1, 2 and 4 kanal), farmhouses
   of 2 to 10 kanal on Bedian, Barki and Raiwind Road, and new areas (DHA Phase 5, Lahore
   Cantt, Lake City).

Every price and listing is made up. Generated listings use real, free-licensed photos of
Pakistani houses from Wikimedia Commons (data/pk_photos.json, downloaded by
scripts/fetch_photos.py). The photos are illustrative and do not show the listed homes.

Run: python scripts/fetch_photos.py && python scripts/generate_listings.py
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "properties.json"
PHOTOS = ROOT / "realestate_agent" / "static" / "homes" / "photos.json"
PK_PHOTOS = ROOT / "data" / "pk_photos.json"
SQFT_PER_MARLA = 225  # The common Lahore convention.
MARLA_PER_KANAL = 20
HAND_WRITTEN = 12
NEW_LISTINGS = 60
WIDE_LISTINGS = 108

# name: (lat, lng, price per sqft PKR, commute minutes to Mall Road, typical features)
AREAS = {
    "Johar Town": (31.4697, 74.2728, 17500, 25, ["school_nearby", "garden", "parking"]),
    "Allama Iqbal Town": (31.5146, 74.2896, 16500, 18, ["parking", "security"]),
    "Model Town": (31.4835, 74.3256, 18000, 12, ["garden", "renovated"]),
    "Park View City": (31.4760, 74.2360, 14000, 38, ["garden", "security", "solar"]),
    "Gulberg": (31.5204, 74.3487, 19500, 9, ["metro_nearby", "security", "elevator"]),
    "DHA Phase 6": (31.4720, 74.4520, 24000, 30, ["security", "garden", "solar", "garage"]),
    "Bahria Town": (31.3685, 74.1824, 15000, 45, ["security", "garden", "school_nearby"]),
    "Wapda Town": (31.4387, 74.2604, 14500, 30, ["school_nearby", "parking"]),
}
# Added with the wider set. Kept apart so the 60 listings above keep their random stream.
NEW_AREAS = {
    "DHA Phase 5": (31.4628, 74.4085, 23500, 26, ["security", "garden", "parking", "solar"]),
    "Lahore Cantt": (31.5120, 74.3750, 21000, 15, ["security", "garden", "parking"]),
    "Lake City": (31.3960, 74.2450, 15500, 42, ["security", "garden", "school_nearby"]),
}
# Farmhouse belts: price per sqft is per plot area, so it is much lower than in town.
FARM_AREAS = {
    "Bedian Road": (31.4350, 74.4950, 5200, 40, ["garden", "security", "orchard", "swimming_pool"]),
    "Barki Road": (31.4750, 74.5200, 4200, 42, ["garden", "orchard", "security"]),
    "Raiwind Road": (31.3750, 74.2700, 3200, 45, ["garden", "orchard", "parking"]),
}
ALL_AREAS = {**AREAS, **NEW_AREAS, **FARM_AREAS}

TYPES = {  # type: marla sizes (used by the 60 earlier listings)
    "house": [5, 7, 10, 10, 12, 20],
    "townhouse": [4, 5, 6],
    "apartment": [3, 4, 5, 6, 8],
}
EXTRA_FEATURES = ["solar", "garage", "garden", "balcony", "parking", "renovated", "needs_renovation",
                  "school_nearby", "security", "elevator", "metro_nearby"]
LUXURY_FEATURES = ["basement", "servant_quarters", "home_theatre", "swimming_pool", "solar", "garage",
                   "garden", "security", "renovated"]
FARM_FEATURES = ["swimming_pool", "orchard", "servant_quarters", "solar", "garage", "basement",
                 "tube_well", "lawn_events", "security"]
DESCRIPTIONS = {
    "house": "{marla} marla {age} house on a {street} street, with {feat}.",
    "townhouse": "{age} townhouse in a gated row, with {feat}.",
    "apartment": "{age} apartment on floor {floor}, with {feat}.",
}
WIDE_DESCRIPTIONS = {
    "house": [
        "{size} {age} house on a {street} street, with {feat}.",
        "{age} double-storey {size} home, {street} plot, with {feat}.",
        "Family house on a {size} {street} plot: {feat}.",
    ],
    "kanal": [
        "{size} {age} bungalow on a {street} plot, with {feat}.",
        "Spacious {size} house with a front lawn, {floors} floors and {feat}.",
        "{age} {size} corner-style home in a gated block, with {feat}.",
    ],
    "farmhouse": [
        "{size} farmhouse with a main house, open lawns and {feat}.",
        "Gated {size} farmhouse set back from the road, with {feat}.",
        "{age} {size} farmhouse with fruit trees, a long driveway and {feat}.",
    ],
    "townhouse": ["{age} townhouse in a gated row, with {feat}."],
    "apartment": ["{age} apartment on floor {floor}, with {feat}."],
}


def size_label(marla: float) -> str:
    """'5 Marla', '1 Kanal', '2.5 Kanal' - the way Lahore listings name plot sizes."""
    if marla >= MARLA_PER_KANAL and marla % 10 == 0:
        kanal = marla / MARLA_PER_KANAL
        return f"{kanal:g} Kanal"
    return f"{marla:g} Marla"


def beds_for(kind: str, marla: int) -> int:
    if kind == "apartment":
        return 1 if marla <= 3 else 2 if marla <= 5 else 3
    return 2 if marla <= 4 else 3 if marla <= 7 else 4 if marla <= 12 else 5


def make(rng: random.Random, n: int) -> dict:
    area = list(AREAS)[n % len(AREAS)] if n < 40 else rng.choice(list(AREAS))
    lat, lng, ppsf, commute, typical = AREAS[area]
    kind = rng.choices(["house", "townhouse", "apartment"],
                       [5, 2, 4 if area in ("Gulberg", "Allama Iqbal Town") else 2])[0]
    marla = rng.choice(TYPES[kind])
    sqft = marla * SQFT_PER_MARLA + rng.randint(-60, 60)
    year = rng.randint(1995, 2025)
    age_factor = 1.08 if year >= 2020 else 0.9 if year < 2005 else 1.0
    price = round(sqft * ppsf * age_factor * rng.uniform(0.9, 1.1), -5)
    features = sorted(set(rng.sample(typical, k=min(2, len(typical))) + rng.sample(EXTRA_FEATURES, 2)))
    if "renovated" in features and "needs_renovation" in features:
        features.remove("renovated" if year < 2005 else "needs_renovation")
    if kind != "apartment":
        features = [f for f in features if f != "elevator"]
    beds = beds_for(kind, marla)
    age = "new-build" if year >= 2022 else "well kept" if year >= 2010 else "older"
    feat = ", ".join(f.replace("_", " ") for f in features[:3])
    return {
        "id": f"P-{101 + HAND_WRITTEN + n}",
        "title": f"{size_label(marla)} {kind} in {area}",
        "area": area,
        "type": kind,
        "price_pkr": int(price),
        "beds": beds,
        "baths": max(1, beds - rng.randint(0, 1)),
        "area_sqft": sqft,
        "year_built": year,
        "status": rng.choices(["available", "under_offer", "sold"], [16, 2, 2])[0],
        "features": features,
        "commute_min_to_center": max(5, commute + rng.randint(-5, 6)),
        "description": DESCRIPTIONS[kind].format(
            marla=marla, age=age, street=rng.choice(["quiet", "wide", "corner", "park-facing"]),
            feat=feat, floor=rng.randint(1, 12)).capitalize(),
    }


# The wider set: (area pool, type, plot sizes in marla, weight). Sizes repeat to set how common they are.
WIDE_MIX = [
    (list(AREAS) + list(NEW_AREAS), "house", [3, 5, 5, 7, 8, 10, 10, 12, 15], 9),
    (["DHA Phase 6", "DHA Phase 5", "Lahore Cantt", "Model Town", "Gulberg", "Bahria Town",
      "Lake City", "Johar Town", "Park View City", "Wapda Town"], "kanal", [20, 20, 20, 30, 40, 40], 7),
    (["DHA Phase 6", "DHA Phase 5", "Lahore Cantt", "Gulberg", "Model Town"], "kanal", [60, 80], 2),
    (list(FARM_AREAS), "farmhouse", [40, 60, 80, 80, 100, 100, 160, 200], 7),
    (["Gulberg", "Lahore Cantt", "Bahria Town", "DHA Phase 5", "Lake City", "Johar Town"], "apartment", [4, 5, 6, 8, 10], 3),
    (["Lake City", "Park View City", "Bahria Town", "Wapda Town", "Johar Town"], "townhouse", [4, 5, 6], 2),
]


def make_wide(rng: random.Random, n: int) -> dict:
    pool, style, sizes, _ = rng.choices(WIDE_MIX, [m[3] for m in WIDE_MIX])[0]
    area = rng.choice(pool)
    lat, lng, ppsf, commute, typical = ALL_AREAS[area]
    marla = rng.choice(sizes)
    kind = "house" if style == "kanal" else style
    sqft = marla * SQFT_PER_MARLA + (rng.randint(-60, 60) if marla < 20 else 0)
    year = rng.randint(1998, 2026)
    age_factor = 1.08 if year >= 2020 else 0.9 if year < 2005 else 1.0
    # 1 kanal plots in good blocks sell at a small premium; very large plots cost less per sq ft.
    size_factor = (1.05 if marla <= 20 else 0.92 if marla <= 40 else 0.8) if style == "kanal" else 1.0
    price = round(sqft * ppsf * age_factor * size_factor * rng.uniform(0.88, 1.12), -5)

    extra = {"kanal": LUXURY_FEATURES, "farmhouse": FARM_FEATURES}.get(style, EXTRA_FEATURES)
    k = 3 if style in ("kanal", "farmhouse") else 2
    features = sorted(set(rng.sample(typical, k=min(2, len(typical))) + rng.sample(extra, k)))
    if "renovated" in features and "needs_renovation" in features:
        features.remove("renovated" if year < 2005 else "needs_renovation")
    if kind != "apartment":
        features = [f for f in features if f not in ("elevator", "balcony")]
    if area not in ("Gulberg", "Allama Iqbal Town", "Lahore Cantt"):
        features = [f for f in features if f != "metro_nearby"]  # Only these areas sit near the Metro.

    if style == "farmhouse":
        beds = 4 if marla <= 40 else 5 if marla <= 80 else 6
    elif style == "kanal":
        beds = 5 if marla <= 20 else 6 if marla <= 40 else 7
    else:
        beds = beds_for(kind, marla)
    age = "new-build" if year >= 2022 else "well kept" if year >= 2010 else "older"
    feat = ", ".join(f.replace("_", " ") for f in features[:3])
    size = size_label(marla)
    title_kind = {"farmhouse": "farmhouse", "kanal": "house"}.get(style, kind)
    title = f"{size} {title_kind} in {area}"
    if style == "farmhouse":
        title = f"{size} farmhouse on {area}"
    elif style == "apartment":
        title = f"{beds} Bed apartment in {area}"
    desc = rng.choice(WIDE_DESCRIPTIONS[style]).format(
        size=size, age=age, street=rng.choice(["quiet", "wide", "corner", "park-facing", "boulevard"]),
        feat=feat, floor=rng.randint(1, 14), floors=rng.choice(["two", "two and a half", "three"]))
    return {
        "id": f"P-{101 + HAND_WRITTEN + NEW_LISTINGS + n}",
        "title": title,
        "area": area,
        "type": kind,
        "price_pkr": int(price),
        "beds": beds,
        "baths": beds + (1 if style in ("kanal", "farmhouse") else -rng.randint(0, 1)),
        "area_sqft": sqft,
        "year_built": year,
        "status": rng.choices(["available", "under_offer", "sold"], [17, 2, 1])[0],
        "features": features,
        "commute_min_to_center": max(5, commute + rng.randint(-5, 6)),
        "description": desc[0].upper() + desc[1:],
    }


def photo_entry(p: dict) -> dict:
    folder = p.get("dir", "pk")  # Older Lahore photos live in homes/lahore and carry their own attribution.
    return {
        "src": f"/static/homes/{folder}/{p['slug']}.jpg",
        "thumb": f"/static/homes/{folder}/{p['slug']}-sm.jpg",
        "alt": p["alt"],
        "credit": p["source_page"],
        "attribution": p.get("attribution") or f"{p['title']} by {p['artist']}, {p['license']}, via Wikimedia Commons",
        "license_url": p["license_url"],
    }


def pool_for(h: dict) -> str:
    if h["type"] == "farmhouse":
        return "farm"
    if h["type"] in ("townhouse", "apartment"):
        return "townhouse"
    return "small" if h["marla"] <= 7.5 else "mid" if h["marla"] < 19 else "kanal"


def assign_photos(homes: list[dict], hand: list[dict], photos: dict) -> dict:
    pk = json.loads(PK_PHOTOS.read_text(encoding="utf-8"))
    pools: dict[str, list[dict]] = {}
    for p in pk:
        pools.setdefault(p["pool"], []).append(p)
    # Interior shots from the hand-written listings (Unsplash), used as a second photo for flats.
    flat_inside = [photos[h["id"]][-1] for h in hand if h["type"] == "apartment" and h["id"] in photos]
    flat_front = [photos[h["id"]][0] for h in hand if h["type"] == "apartment" and h["id"] in photos]
    for i, h in enumerate(homes[HAND_WRITTEN:]):
        pool = pools[pool_for(h)]
        # Scatter by a stable hash of the id, so homes next to each other in a price sort rarely share a photo.
        pick = random.Random("photo" + h["id"]).randrange(len(pool))
        main = pool[pick]
        second = pool[(pick + 1 + i % (len(pool) - 1)) % len(pool)]
        gallery = [photo_entry(main)]
        if second is not main:
            gallery.append(photo_entry(second))
        if h["type"] == "farmhouse":
            gallery.append(photo_entry(pools["farm_in"][0]))
            gallery.append(photo_entry(pools["garden"][i % len(pools["garden"])]))
        elif h["marla"] >= 10 and h["type"] == "house":
            gallery.append(photo_entry(pools["garden"][i % len(pools["garden"])]))
        if h["type"] == "apartment" and flat_front:
            # Few free photos of Pakistani flats exist, so half the flats lead with a stock building shot.
            gallery.append(flat_inside[pick % len(flat_inside)])
            if pick % 2 == 0:
                gallery.insert(0, flat_front[(pick // 2) % len(flat_front)])
        photos[h["id"]] = gallery
    return photos


def main() -> None:
    hand = json.loads(DATA.read_text(encoding="utf-8"))[:HAND_WRITTEN]
    rng = random.Random(27)
    homes = hand + [make(rng, n) for n in range(NEW_LISTINGS)]
    wide = random.Random(38)
    homes += [make_wide(wide, n) for n in range(WIDE_LISTINGS)]
    for h in homes:
        lat, lng = ALL_AREAS[h["area"]][:2]
        h["marla"] = round(h["area_sqft"] / SQFT_PER_MARLA, 1)
        jitter = random.Random(h["id"])  # Stable per listing.
        spread = 0.03 if h["area"] in FARM_AREAS else 0.012
        h["lat"] = round(lat + jitter.uniform(-spread, spread), 5)
        h["lng"] = round(lng + jitter.uniform(-spread, spread), 5)
    DATA.write_text(json.dumps(homes, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    photos = json.loads(PHOTOS.read_text(encoding="utf-8"))
    photos = {k: v for k, v in photos.items() if k in {h["id"] for h in hand}}
    assign_photos(homes, hand, photos)
    PHOTOS.write_text(json.dumps(photos, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    kinds: dict[str, int] = {}
    for h in homes:
        kinds[h["type"]] = kinds.get(h["type"], 0) + 1
    print(f"{len(homes)} listings across {len({h['area'] for h in homes})} areas: {kinds}")


if __name__ == "__main__":
    main()
