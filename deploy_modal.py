import modal
import os
import shutil


# --- 1. TẠO VOLUMES TỰ ĐỘNG NẾU CHƯA TỒN TẠI ---
models_volume = modal.Volume.from_name(
    "my-quantized-models-volume", 
    create_if_missing=True  # ✅ Tự động tạo nếu chưa có
)

torch_cache_vol = modal.Volume.from_name(
    "torch-cache-volume",
    create_if_missing=True  # ✅ Tự động tạo nếu chưa có
)

triton_cache_vol = modal.Volume.from_name(
    "triton-cache-volume",
    create_if_missing=True  # ✅ Tự động tạo nếu chưa có
)

vllm_compile_cache_vol = modal.Volume.from_name(
    "vllm-compile-cache-volume",
    create_if_missing=True
)

TORCH_CACHE_PATH = "/cache/torch"
TRITON_CACHE_PATH = "/cache/triton"
VLLM_COMPILE_CACHE_PATH = "/cache/vllm_compile"
local_model_dir_path = "./models" 


# --- 2. ĐỊNH NGHĨA MÔI TRƯỜNG APP ---
app_image = (
     modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", 
        add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(requirements=["pyproject.toml"])
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:$PATH",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
        "VLLM_USE_FLASHINFER": "1",   # (giữ flashinfer, nếu không muốn: dùng "0")
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9",  # A10G, A100 tương thích
        "TORCHINDUCTOR_CACHE_DIR": "/cache/torch",
        "TRITON_CACHE_DIR": "/cache/triton",
        "MODEL_DIR": "/models_dir",
        "VLLM_EAGER": "1", 
    })
    .run_commands(
        "mkdir -p /cache/torch",
        "mkdir -p /cache/triton",
        "nvcc --version",
    )
    .add_local_dir(".", "/app",
                    ignore=[ 
            ".venv",
            ".venv/**",
            "**/.venv/**",
            "__pycache__",
            "**/__pycache__",
            "**/*.pyc",
            "**/*.pyo",
            ".git",
            ".git/**",
            "models",
            "models/**",
            "*.md",
            "*.local",
            "start.sh",
        ]
    )
)



# --- 3. ĐỊNH NGHĨA APP ---
app = modal.App(name="my-vllm-fastapi-app")


# --- 4. HÀM UPLOAD ---
upload_image = modal.Image.debian_slim().add_local_dir(
    local_model_dir_path, 
    remote_path="/local_models_ro"
)

@app.function(
    image=upload_image,
    volumes={"/models_dir": models_volume},
    timeout=1800,
)
def upload_local_models():
    src_path = "/local_models_ro"
    dest_path = "/models_dir"
    
    check_file_name = "language_model/GRPO-Vi-Qwen2-7B-RAG-W4A16/config.json"
    check_file_on_volume = os.path.join(dest_path, check_file_name)
    
    if os.path.exists(check_file_on_volume):
        print(f"File {check_file_on_volume} đã tồn tại. Bỏ qua upload.")
        return

    print(f"Đang copy models từ '{src_path}' sang '{dest_path}'...")

    try:
        shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
    except Exception as e:
        print(f"Lỗi: {e}")
        return

    print("Đang commit volume...")
    models_volume.commit()
    print("✅ Upload thành công!")


# --- 5. WEB ENDPOINT ---
@app.function(
    image=app_image,
    volumes={
        "/models_dir": models_volume,
        TORCH_CACHE_PATH: torch_cache_vol,
        TRITON_CACHE_PATH: triton_cache_vol,
        VLLM_COMPILE_CACHE_PATH: vllm_compile_cache_vol,
    },
    gpu="A10G",
    timeout=3600,
    keep_warm=0,
    container_idle_timeout=300,
    scaledown_window=120,

)
@modal.asgi_app()
def run_fastapi_app():
    import sys
    sys.path.insert(0, "/app")
    from main import app as fastapi_app
    return fastapi_app
