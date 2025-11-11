# from fastapi import FastAPI
# # from fastapi.middleware.cors import CORSMiddleware
# from api.v1 import routes as v1_routes
# from contextlib import asynccontextmanager
# from services.vllm_service import vLLMService
# from vllm.engine.async_llm_engine import AsyncLLMEngine
# # from dotenv import load_dotenv


# # load_dotenv()
# ready: bool = False


# ###############################################################################
# #
# #          Manage resources with lifespan for safe shutdown and cleanup
# #
# ###############################################################################
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     vllm_engine: AsyncLLMEngine = await vLLMService.init_resource()
#     app.state.vllm_engine = vllm_engine
#     global ready
#     ready = True
#     try:
#         yield
#     finally:
#         ready = False


# ###############################################################################
# #
# #                      Initialize FastAPI application
# #
# ###############################################################################
# app = FastAPI(title="AI Chatbot API", lifespan=lifespan)

# # CORS setup for Streamlit frontend
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["http://localhost:8501"],
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # Include versioned routers
# app.include_router(v1_routes.router, prefix="/api/v1")


# @app.get("/ping")
# def health_check():
#     return {"": 204} if not ready else {"status": "healthy"}

import signal
from fastapi import FastAPI, status, Request
from fastapi.responses import JSONResponse, StreamingResponse
from api.v1 import routes as v1_routes
from contextlib import asynccontextmanager
from services.vllm_service import vLLMService
from vllm.engine.async_llm_engine import AsyncLLMEngine

ready: bool = False
engine: AsyncLLMEngine | None = None


###############################################################################
#          Manage resources with lifespan for safe startup and cleanup
###############################################################################
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ready, engine
    print("🚀 Application starting up (lazy vLLM loading)...")
    ready = False

    if engine is None:  # ✅ chỉ preload 1 lần
        try:
            from services.vllm_service import vLLMService
            print("⚙️ Preloading vLLM engine...")
            engine = await vLLMService.init_resource()
            app.state.vllm_engine = engine
            ready = True
            print("✅ vLLM engine ready!")
        except Exception as e:
            print(f"❌ Failed to preload engine: {e}")
    else:
        print("♻️ Engine already initialized, skipping preload...")

    yield
    print("🛑 Application shutting down...")



###############################################################################
#                      Initialize FastAPI application
###############################################################################
app = FastAPI(
    title="AI Chatbot API",
    version="1.0.0",
    lifespan=lifespan
)

# Include versioned routers
app.include_router(v1_routes.router, prefix="/api/v1")


###############################################################################
#                      Dynamic model loader (auto-reconnect)
###############################################################################
async def get_engine(app: FastAPI) -> AsyncLLMEngine:
    """
    Return active vLLM engine. Auto-init or reconnect if needed.
    """
    global ready, engine

    if engine is None:
        print("⚙️ (Re)initializing vLLM engine...")
        try:
            engine = await vLLMService.init_resource()
            app.state.vllm_engine = engine
            ready = True
            print("✅ vLLM engine ready!")
        except Exception as e:
            print(f"❌ [Init Error] Failed to start vLLM engine: {e}")
            ready = False
            raise

    return engine


###############################################################################
#                           Health Check Endpoints
###############################################################################
@app.get("/")
async def root():
    return {
        "name": "AI Chatbot API",
        "version": "1.0.0",
        "status": "running" if ready else "starting"
    }


@app.get("/health")
async def health():
    if not ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unavailable",
                "ready": False,
                "message": "vLLM engine is still loading"
            }
        )
    return {
        "status": "healthy",
        "ready": True,
        "service": "vllm",
        "model": "GRPO-Vi-Qwen2-7B-RAG-W4A16"
    }


@app.get("/ready")
async def readiness():
    if not ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"ready": False}
        )
    return {"ready": True}


@app.get("/live")
async def liveness():
    return {"alive": True}


@app.get("/ping")
async def ping():
    if not ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "starting"}
        )
    return {"status": "healthy"}


###############################################################################
#                   Safe generate endpoint with retry logic
###############################################################################
@app.post("/api/v1/generate")
async def generate(request: Request):
    """
    Generate endpoint with auto-reconnect logic for vLLM engine.
    """
    global engine, ready
    data = await request.json()
    prompt = data.get("prompt", "")

    try:
        llm_engine = await get_engine(app)
        stream = vLLMService.generate_answer(llm_engine, prompt)
        return StreamingResponse(stream, media_type="text/event-stream")

    except Exception as e:
        print(f"⚠️ [Engine error] {e} — retrying re-init ...")
        # Reconnect logic nếu engine bị mất do scale-down
        try:
            engine = await vLLMService.init_resource()
            ready = True
            print("🔁 vLLM engine restarted successfully!")
            stream = vLLMService.generate_answer(engine, prompt)
            return StreamingResponse(stream, media_type="text/event-stream")
        except Exception as e2:
            print(f"💥 [Critical] Could not recover engine: {e2}")
            ready = False
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Model unavailable, please retry later."}
            )

