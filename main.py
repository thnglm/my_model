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

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from api.v1 import routes as v1_routes
from contextlib import asynccontextmanager
from services.vllm_service import vLLMService
from vllm.engine.async_llm_engine import AsyncLLMEngine

ready: bool = False

###############################################################################
#          Manage resources with lifespan for safe shutdown and cleanup
###############################################################################
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager - load vLLM engine on startup
    """
    global ready
    
    print("🚀 Application starting up - Loading vLLM engine...")
    try:
        vllm_engine: AsyncLLMEngine = await vLLMService.init_resource()
        app.state.vllm_engine = vllm_engine
        ready = True
        print("✅ vLLM engine loaded and ready!")
    except Exception as e:
        print(f"❌ Failed to load vLLM engine: {e}")
        ready = False
        raise  # Re-raise để Modal biết có lỗi
    
    yield  # App đang chạy
    
    # Cleanup khi shutdown
    print("🛑 Application shutting down...")
    ready = False

###############################################################################
#                      Initialize FastAPI application
###############################################################################
app = FastAPI(
    title="AI Chatbot API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS setup (nếu cần)
# from fastapi.middleware.cors import CORSMiddleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:8501"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# Include versioned routers
app.include_router(v1_routes.router, prefix="/api/v1")

###############################################################################
#                           Health Check Endpoints
###############################################################################

@app.get("/")
async def root():
    """Root endpoint - basic info"""
    return {
        "name": "AI Chatbot API",
        "version": "1.0.0",
        "status": "running" if ready else "starting"
    }

@app.get("/health")
async def health():
    """
    Comprehensive health check - checks if app is ready
    Use this for readiness probes
    """
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
    """
    Readiness probe - Kubernetes/Modal style
    Returns 200 only when ready to serve requests
    """
    if not ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"ready": False}
        )
    return {"ready": True}

@app.get("/live")
async def liveness():
    """
    Liveness probe - checks if app is alive (not frozen)
    Always returns 200 if FastAPI is responding
    """
    return {"alive": True}

# Giữ lại endpoint cũ để backward compatible
@app.get("/ping")
async def ping():
    """Legacy health check endpoint"""
    if not ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "starting"}
        )
    return {"status": "healthy"}
