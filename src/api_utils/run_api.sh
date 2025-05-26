export CUDA_VISIBLE_DEVICES=0 
export MODEL_DIR="Tongyi-Zhiwen/QwenLong-CPRS-7B"
uvicorn run_api:app --port 8091 --host '0.0.0.0' --workers 1
