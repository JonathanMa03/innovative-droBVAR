def get_device(prefer_mps: bool = True):
    """
    Return the best available PyTorch device.

    Priority:
    1. CUDA
    2. MPS on Apple Silicon
    3. CPU
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return torch.device("cuda")

    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")