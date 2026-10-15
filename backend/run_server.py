import sys
from pathlib import Path

# Add backend directory to path
backend_dir = str(Path(__file__).resolve().parent)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import uvicorn

if __name__ == "__main__":
    print("Starting FastAPI backend server on http://0.0.0.0:8000...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
