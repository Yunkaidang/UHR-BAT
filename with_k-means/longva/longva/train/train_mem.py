import torch.multiprocessing as mp
import torch

from longva.train.train import train


def _patch_torch_vector_norm_for_int():
    """
    临时修补：某些 DeepSpeed 版本在计算全局梯度范数时，
    可能将整型标量拼进 stack，导致 torch.linalg.vector_norm 接收到 Long dtype。
    这里将整型张量在进入 vector_norm 前强制转换为 float，以避免崩溃。
    """
    try:
        # 1) patch torch.linalg.vector_norm
        orig_vec_norm = torch.linalg.vector_norm

        def _safe_vec_norm(x, *args, **kwargs):
            if isinstance(x, torch.Tensor) and not x.is_floating_point():
                x = x.float()
            return orig_vec_norm(x, *args, **kwargs)

        torch.linalg.vector_norm = _safe_vec_norm  # type: ignore[attr-defined]

        # 2) patch torch.norm（部分调用直接走 torch.norm）
        orig_norm = torch.norm

        def _safe_norm(x, *args, **kwargs):
            if isinstance(x, torch.Tensor) and not x.is_floating_point():
                x = x.float()
            return orig_norm(x, *args, **kwargs)

        torch.norm = _safe_norm  # type: ignore[assignment]

        # 3) patch torch.functional.norm（某些路径会直接调用 functional.norm）
        import torch.functional as F

        orig_f_norm = F.norm

        def _safe_f_norm(x, *args, **kwargs):
            if isinstance(x, torch.Tensor) and not x.is_floating_point():
                x = x.float()
            return orig_f_norm(x, *args, **kwargs)

        F.norm = _safe_f_norm  # type: ignore[assignment]
    except Exception:
        # 如果补丁失败，不影响后续流程
        pass


if __name__ == "__main__":
    _patch_torch_vector_norm_for_int()
    mp.set_start_method("spawn", force=True)
    train()
