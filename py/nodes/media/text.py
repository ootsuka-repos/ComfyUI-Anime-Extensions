"""Run the text and vision backend inside the ComfyUI queue."""
from __future__ import annotations

import asyncio
import json

from comfy import model_management
from comfy_api.latest import io

from .text_backend import run_openai_agent


class TextCompletion(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ComfyUIExtensions.TextCompletion",
            display_name="Text Completion",
            category="ComfyUIExtensions/Text",
            inputs=[io.String.Input("request_json", multiline=True, default='{"user_prompt":"","system_prompt":"","log_tag":"text"}')],
            outputs=[io.String.Output("text")],
            is_output_node=True,
            not_idempotent=True,
        )

    @classmethod
    async def execute(cls, request_json):
        request = json.loads(request_json)
        allowed = {"user_prompt", "system_prompt", "log_tag", "model", "response_format",
                   "image_data_urls", "temperature", "repetition_penalty", "min_p"}
        if not isinstance(request, dict) or set(request) - allowed:
            raise ValueError("Invalid text request fields")
        if any(not isinstance(request.get(k), str) for k in ("user_prompt", "system_prompt", "log_tag")):
            raise ValueError("Text request requires user_prompt, system_prompt and log_tag strings")
        # URL, credentials and model lifecycle configuration belong to the host.
        # Never expose the backend's tool execution through a workflow input.
        task = asyncio.create_task(run_openai_agent(**request, allowed_tools=[]))
        try:
            while not task.done():
                model_management.throw_exception_if_processing_interrupted()
                await asyncio.wait({task}, timeout=0.2)
            result = task.result()
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        return io.NodeOutput(result.text, ui={"text": [result.text]})


class TextModelRelease(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ComfyUIExtensions.TextModelRelease",
            display_name="Release Local Text Model",
            category="ComfyUIExtensions/Text",
            inputs=[], outputs=[io.Boolean.Output("released")],
            is_output_node=True, not_idempotent=True,
        )

    @classmethod
    async def execute(cls):
        from .text_backend import release_local_model_for_gpu
        released = await asyncio.to_thread(release_local_model_for_gpu)
        return io.NodeOutput(released, ui={"text": [json.dumps(released)]})
