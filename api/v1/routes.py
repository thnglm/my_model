from fastapi import APIRouter, Request, Depends
from fastapi.responses import StreamingResponse
from vllm.engine.async_llm_engine import AsyncLLMEngine
from api.v1.models import ChatRequest
from services.vllm_service import vLLMService

router = APIRouter()


###############################################################################
#
#                     Dependency Injection for FastAPI
#
###############################################################################
def get_vllm_engine(
    request: Request
) -> AsyncLLMEngine:
    return request.app.state.vllm_engine


###############################################################################
#
#                               API Endpoints
#
###############################################################################
@router.post("/ask")
async def ask(
    request: ChatRequest,
    vllm_engine: AsyncLLMEngine = Depends(
        get_vllm_engine
    )
) -> StreamingResponse:
    question = request.question.strip()
    return StreamingResponse(
        vLLMService.generate_answer(
            llm_engine=vllm_engine,
            prompt=question
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # "Access-Control-Allow-Origin": "*",  # For CORS if needed
        }
    )
