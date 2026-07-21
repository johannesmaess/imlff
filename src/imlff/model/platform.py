import sys
import time
import torch

def wait_for_cuda(timeout=30):
    if sys.platform == 'darwin':
        return 'cpu'
    start = time.time()
    while time.time() - start < timeout:
        if torch.cuda.is_available():
            return 'cuda'
        print("Waiting for CUDA to become available...")
        time.sleep(2)
    print("CUDA not available after timeout, using CPU.")
    return 'cpu'