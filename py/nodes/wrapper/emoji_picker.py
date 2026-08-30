from comfy_api.latest import io
from .common import PACKAGE_NAME, CATEGORY
from ...node_utils import mk_name


class EmojiPicker(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=mk_name(PACKAGE_NAME, "EmojiPicker"), 
            display_name="IrodoriTTS Emoji Picker", 
            category=CATEGORY, 
            inputs=[], 
            outputs=[], 
        )
    
    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput()

