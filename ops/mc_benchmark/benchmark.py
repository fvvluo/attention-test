import torch

import cutlass.cute as cute

def benchmark_kernel(kernel, *args, warmup: int = 10, iterations: int = 100) -> float:
    for _ in range(warmup):
        kernel(*args)

    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing = True)
    end_event = torch.cuda.Event(enable_timing = True)

    start_event.record()

    for _ in range(iterations):
        kernel(*args)

    end_event.record()

    end_event.synchronize()

    total_time = start_event.elapsed_time(end_event)
    average_time = total_time / iterations

    return average_time