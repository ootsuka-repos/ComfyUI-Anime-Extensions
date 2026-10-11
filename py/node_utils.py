NODE_NAMESPACE = "ComfyUIExtensions"

def mk_name(*args):
    parts = [NODE_NAMESPACE] + list(args)
    return ".".join(parts)

def mk_category(*args):
    parts = [NODE_NAMESPACE] + list(args)
    return "/".join(parts)


