import torch.utils.checkpoint as cp

try:
    from torch.utils import _pytree as _torch_pytree

    if not hasattr(_torch_pytree, "register_pytree_node") and hasattr(_torch_pytree, "_register_pytree_node"):
        def register_pytree_node(*args, **kwargs):
            return _torch_pytree._register_pytree_node(*args, **kwargs)

        _torch_pytree.register_pytree_node = register_pytree_node
except Exception:
    # 仅作为兼容补丁，失败时继续走下去
    pass


# 备份原来的两个函数
_orig_checkpoint = cp.checkpoint
_orig_checkpoint_sequential = cp.checkpoint_sequential


# 用 non-reentrant（只会一次 forward + 一次 backward）
def _no_reentrant_checkpoint(fn, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return _orig_checkpoint(fn, *args, **kwargs)


def _no_reentrant_checkpoint_seq(functions, segments, *inputs, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return _orig_checkpoint_sequential(functions, segments, *inputs, **kwargs)


# 覆盖全局
cp.checkpoint = _no_reentrant_checkpoint
cp.checkpoint_sequential = _no_reentrant_checkpoint_seq

from .model import LlavaLlamaForCausalLM
