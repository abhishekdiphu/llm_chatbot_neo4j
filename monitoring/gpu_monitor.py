
import time
import torch
class GPUMemoryMonitor:
    def __init__(self, update_interval=5):
        self.update_interval = update_interval

    def get_gpu_memory_info():
        """Get current GPU memory usage in MB for precision with small allocations."""
        if not torch.mps.is_available():
            return {"allocated_mb": 0, "reserved_mb": 0, "peak_mb": 0, "total_mb": 0, "free_mb": 0}
        torch.mps.synchronize()
        # MPS reports bytes
        allocated = torch.mps.current_allocated_memory()
        driver_allocated = torch.mps.driver_allocated_memory()
        recommended_max = torch.mps.recommended_max_memory()
        allocated = allocated / (1024 ** 2)
        driver_allocated = driver_allocated / (1024 ** 2)
        recommended_max = recommended_max / (1024 ** 2)
        return {
            "allocated_mb": round(allocated, 1),
            "reserved_mb": round(driver_allocated, 1),
            "peak_mb": 0,
            "total_mb": round(recommended_max, 1),
            "free_mb": round(max(recommended_max - driver_allocated, 0),1)
        }

    def print_gpu_memory(self, label=""):
        """Print current GPU memory status."""
        info = self.get_gpu_memory_info()
        print(f"{label}")
        print(f"  Allocated: {info['allocated_mb']:,.1f} MB  ({info['allocated_mb']/1024:.2f} GB)")
        print(f"  Reserved:  {info['reserved_mb']:,.1f} MB  ({info['reserved_mb']/1024:.2f} GB)")
        print(f"  Peak:      {info['peak_mb']:,.1f} MB  ({info['peak_mb']/1024:.2f} GB)")
        print(f"  Free:      {info['free_mb']:,.1f} MB  ({info['free_mb']/1024:.2f} GB)")
        return info

# Check current memory state

if __name__ == "__main__":
    update_time = 5
    monitor = GPUMemoryMonitor(5)
    while True:
      monitor.print_gpu_memory("Current GPU Memory:")
      time.sleep(update_time) 