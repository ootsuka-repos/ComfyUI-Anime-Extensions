from pathlib import Path

import folder_paths
from comfy_api.latest import io

from ...node_utils import mk_name
from .common import CATEGORY, PACKAGE_NAME
from .irodori_common import IO_LORA_STACK, resolve_lora_path


class IrodoriLoRAStack(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        lora_files = ["None"] + folder_paths.get_filename_list("loras")

        return io.Schema(
            node_id=mk_name(PACKAGE_NAME, "LoRAStack"),
            display_name="IrodoriTTS LoRA Stack",
            category=CATEGORY,
            inputs=[
                IO_LORA_STACK.Input(
                    "prev",
                    optional=True,
                    tooltip="前段のLoRA Stackを接続します。複数LoRAを積む場合に使用します。",
                ),
                io.Combo.Input(
                    "lora",
                    options=lora_files,
                    tooltip="追加するIrodoriTTS用LoRAアダプタを選択します。",
                ),
                io.Float.Input(
                    "strength",
                    min=-10.0,
                    max=10.0,
                    step=0.01,
                    default=1.0,
                    tooltip="LoRAの適用強度です。1.0がLoRA本来の強さ、0.0で無効相当です。",
                ),
            ],
            outputs=[
                IO_LORA_STACK.Output(display_name="irodori_lora_stack"),
            ],
        )

    @classmethod
    def execute(
        cls,
        lora: str,
        strength: float,
        prev: list | None = None,
    ):
        stack = list(prev or [])
        path = resolve_lora_path(lora)
        if path:
            stack.append(
                {
                    "path": path,
                    "strength": float(strength),
                    "name": Path(path).name,
                }
            )
        return io.NodeOutput(stack)
