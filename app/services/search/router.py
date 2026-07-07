from fastapi import APIRouter
from typing import List
from app.services.search.engine import get_search_engine
from app.shared.schemas import SearchResult

router = APIRouter(prefix="/search", tags=["Search"])


@router.get("", response_model=List[SearchResult])
def search(q: str, top_k: int = 5):
    return get_search_engine().search(q, top_k=top_k)


@router.post("/refresh")
def refresh_index():
    get_search_engine().refresh()
    return {"status": "index refreshed"}
