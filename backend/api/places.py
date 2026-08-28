"""Place picker: search TIGER county subdivisions and fetch one boundary."""

from fastapi import APIRouter, HTTPException, Query

from core import places
from models.schemas import PlaceDetail, PlaceSummary

router = APIRouter(prefix="/api", tags=["places"])


@router.get("/places", response_model=list[PlaceSummary])
def list_places(
    state: str | None = Query(default=None, max_length=2),
    q: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    try:
        return places.query_places(state=state, q=q, limit=limit, offset=offset)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/places/{geoid}", response_model=PlaceDetail)
def place_detail(geoid: str):
    try:
        place = places.get_place(geoid)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    if place is None:
        raise HTTPException(status_code=404, detail=f"No place with GEOID {geoid}.")
    return place
