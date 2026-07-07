from fastapi import APIRouter
from app.services.assistant.engine import get_assistant_engine
from app.shared.schemas import AssistantTurn, AssistantReply

router = APIRouter(prefix="/assistant", tags=["Shopping Assistant"])


@router.post("/chat", response_model=AssistantReply)
def chat(turn: AssistantTurn):
    return get_assistant_engine().handle_turn(turn)
