# Sol-H3-Spark Qwen worker, preserving its pinned NGC Torch and ComfyUI.
FROM nvcr.io/nvidia/pytorch:25.11-py3@sha256:417cbf33f87b5378849df37983552cd1f8bc8b62fe1ceabe004de816a55dff21
ARG COMFYUI_COMMIT=b1693ecba9f5b65f8c80ab36b195ab963ec92413
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN git clone --filter=blob:none --no-checkout https://github.com/Comfy-Org/ComfyUI.git /opt/ComfyUI \
    && git -C /opt/ComfyUI checkout --detach ${COMFYUI_COMMIT} \
    && test "$(git -C /opt/ComfyUI rev-parse HEAD)" = "${COMFYUI_COMMIT}"
RUN sed -E '/^(torch|torchvision|torchaudio)([<=>~! ].*)?$/d' \
      /opt/ComfyUI/requirements.txt > /tmp/comfy-requirements.txt \
    && python -m pip install -r /tmp/comfy-requirements.txt
WORKDIR /opt/ComfyUI
ENTRYPOINT []
CMD ["python"]

# NGC 25.11 omits torchaudio and predates TorchAudio 2.10's stable Torch C++ API.
# Compile the preceding audio source against the preserved NGC Torch headers.
# Qwen uses text/vision inputs; audio samples are encoded by Stage1's audio VAE.
RUN git clone --depth 1 --filter=blob:none --sparse --branch v2.9.1 https://github.com/pytorch/audio.git /opt/torchaudio \
    && git -C /opt/torchaudio sparse-checkout set src tools third_party \
    && test "$(git -C /opt/torchaudio rev-parse HEAD)" = "a224ab24a7f4797f6707051257265e223e12576f"
RUN git -C /opt/torchaudio sparse-checkout add cmake
RUN cd /opt/torchaudio \
    && BUILD_VERSION=2.9.1+nv25.11 USE_CUDA=0 BUILD_SOX=0 USE_FFMPEG=0 MAX_JOBS=8 python -m pip install --no-deps --no-build-isolation -v . \
    && python -c "import torch, torchaudio; assert torch.__version__ == '2.10.0a0+b558c986e8.nv25.11'; print(torchaudio.__version__)"
