from __future__ import annotations

import torch


status = {
    "torch": torch.__version__,
    "torch_built_cuda": torch.version.cuda,
    "available": torch.cuda.is_available(),
    "count": torch.cuda.device_count(),
    "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "free_total": torch.cuda.mem_get_info() if torch.cuda.is_available() else None,
}
print(status)
if not status["available"]:
    raise SystemExit(
        "CUDA is not visible: Windows must report the NVIDIA GPU as Connected/Started before M4.2 can run."
    )
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible in this terminal; M4.2 formal training was not started")
