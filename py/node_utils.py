from functools import wraps
from pathlib import Path

NODE_NAMESPACE = "ComfyUIExtensions"
ROOT_DIR = Path(__file__).parent.parent

def mk_name(*args):
    parts = [NODE_NAMESPACE] + list(args)
    return ".".join(parts)

def mk_category(*args):
    parts = [NODE_NAMESPACE] + list(args)
    return "/".join(parts)


class Endpoint:
    @classmethod
    def _routes(cls):
        from server import PromptServer

        return PromptServer.instance.routes
    
    @classmethod
    def _endpoint(cls, *args):
        parts = [NODE_NAMESPACE] + list(args)
        path = "/".join(parts)
        return f"/{path}"
    
    @classmethod
    def get(cls, *args):
        """GETリクエスト用デコレータ"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            cls._routes().get(cls._endpoint(*args))(wrapper)
            return wrapper
        return decorator
    
    @classmethod
    def post(cls, *args):
        """POSTリクエスト用デコレータ"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            cls._routes().post(cls._endpoint(*args))(wrapper)
            return wrapper
        return decorator


