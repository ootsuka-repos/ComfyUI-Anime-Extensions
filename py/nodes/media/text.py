"""Run the text and vision backend inside the ComfyUI queue."""
from __future__ import annotations

import asyncio
import json

from comfy import model_management
from comfy_api.latest import io


class TextRequest(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ComfyUIExtensions.TextRequest",
            display_name="Text Request",
            category="ComfyUIExtensions/Text",
            description="通常の文章から生成リクエストを作成します。request_jsonをText Completionに接続してください。",
            inputs=[
                io.String.Input(
                    "user_prompt", multiline=True, default="",
                    tooltip="生成してほしい内容や質問をそのまま入力します。JSONを書く必要はありません。",
                ),
                io.String.Input(
                    "system_prompt", multiline=True, default="",
                    tooltip="応答の役割・文体などの指示です。不要なら空欄にします。",
                ),
                io.String.Input(
                    "model", default="", optional=True,
                    tooltip="使用するモデル名です。空欄または空白のみの場合はホスト設定の既定モデルを使用します。",
                ),
            ],
            outputs=[io.String.Output("request_json", tooltip="Text Completionのrequest_jsonに接続します。")],
        )

    @classmethod
    def execute(cls, user_prompt, system_prompt, model=""):
        request = {"user_prompt": user_prompt, "system_prompt": system_prompt, "log_tag": "text"}
        if model and model.strip():
            request["model"] = model.strip()
        return io.NodeOutput(json.dumps(request, ensure_ascii=False))


class TextCompletion(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ComfyUIExtensions.TextCompletion",
            display_name="Text Completion",
            category="ComfyUIExtensions/Text",
            description=(
                "Text Requestのrequest_jsonを接続して文章を生成します。"
                "高度な設定やプログラムからの利用では、従来どおりJSONを直接入力できます。"
                "接続先URL・認証情報はホスト側で設定します。"
            ),
            inputs=[io.String.Input(
                "request_json", multiline=True,
                default='{"user_prompt":"","system_prompt":"","log_tag":"text"}',
                tooltip="Text Requestの出力を接続するか、user_prompt・system_prompt・log_tagを含むJSONを入力します。",
            )],
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
        from .text_backend import run_openai_agent

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
