import os
import sys
import shutil
import subprocess
import json
import urllib.request
import ctypes

def discover():
    print("=== ENVIRONMENT & OLLAMA DISCOVERY ===")
    print("Python executable:", sys.executable)
    print("Python version:", sys.version)
    print("Current working dir:", os.getcwd())

    # 1. Check Ollama on PATH
    ollama_path = shutil.which("ollama")
    print("Ollama on PATH:", ollama_path)

    # 2. Check common Windows install locations
    common_locs = [
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
        os.path.expandvars(r"%PROGRAMFILES%\Ollama\ollama.exe"),
        os.path.expandvars(r"%PROGRAMFILES(X86)%\Ollama\ollama.exe"),
        r"C:\Users\Viknesh G\AppData\Local\Programs\Ollama\ollama.exe",
    ]
    found_exe = None
    for loc in common_locs:
        if os.path.exists(loc):
            print("Found Ollama executable at:", loc)
            found_exe = loc
            break

    exe_to_use = ollama_path or found_exe
    if exe_to_use:
        try:
            ver = subprocess.run([exe_to_use, "--version"], capture_output=True, text=True, timeout=5)
            print("Ollama CLI version:", ver.stdout.strip() or ver.stderr.strip())
        except Exception as e:
            print("Error running ollama --version:", e)

    # 3. Check Local API http://localhost:11434
    api_available = False
    try:
        req = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        data = json.loads(req.read().decode("utf-8"))
        api_available = True
        print("Ollama API reachable at http://localhost:11434")
        models = data.get("models", [])
        print(f"Installed models count: {len(models)}")
        for m in models:
            name = m.get("name")
            size_gb = m.get("size", 0) / (1024**3)
            details = m.get("details", {})
            family = details.get("family", "")
            param_size = details.get("parameter_size", "")
            quant = details.get("quantization_level", "")
            print(f" - Model: {name} | Size: {size_gb:.2f} GB | Family: {family} | Params: {param_size} | Quant: {quant}")
    except Exception as e:
        print("Ollama API not reachable or error:", e)

    # If API not reachable but exe exists, check if 'ollama list' works
    if not api_available and exe_to_use:
        try:
            lst = subprocess.run([exe_to_use, "list"], capture_output=True, text=True, timeout=5)
            print("CLI ollama list output:\n", lst.stdout)
        except Exception as e:
            print("Error running ollama list:", e)

    # 4. Check RAM and Disk using standard library
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        total_ram = stat.ullTotalPhys / (1024**3)
        avail_ram = stat.ullAvailPhys / (1024**3)
        print(f"RAM Total: {total_ram:.2f} GB, Available: {avail_ram:.2f} GB")
    except Exception as e:
        print("RAM check error:", e)

    try:
        total_d, used_d, free_d = shutil.disk_usage("D:\\")
        print(f"Disk D: Total: {total_d / (1024**3):.2f} GB, Free: {free_d / (1024**3):.2f} GB")
        total_c, used_c, free_c = shutil.disk_usage("C:\\")
        print(f"Disk C: Total: {total_c / (1024**3):.2f} GB, Free: {free_c / (1024**3):.2f} GB")
    except Exception as e:
        print("Disk check error:", e)

    # 5. Check GPU
    try:
        import torch
        print("PyTorch version:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("Device count:", torch.cuda.device_count())
            print("Device name:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("Torch check error:", e)

    try:
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        print("nvidia-smi available:", smi.returncode == 0)
        if smi.returncode == 0:
            lines = smi.stdout.splitlines()[:12]
            print("\n".join(lines))
    except Exception:
        print("nvidia-smi not available")

    # 6. Check Python packages
    try:
        import ollama
        print("Python package 'ollama' is installed!")
    except ImportError:
        print("Python package 'ollama' is NOT installed.")

    try:
        import httpx
        print("Python package 'httpx' is installed!")
    except ImportError:
        print("Python package 'httpx' is NOT installed.")

    try:
        import requests
        print("Python package 'requests' is installed!")
    except ImportError:
        print("Python package 'requests' is NOT installed.")

if __name__ == "__main__":
    discover()
