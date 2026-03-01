"""
Root entrypoint for the Evidence-Aware RAG FastAPI application.

This file exposes `app` at the module level so you can run the server with:

    uvicorn main:app --host 0.0.0.0 --port 8000

and is also compatible with platforms that expect `main.py` as the root.
"""

import os

import uvicorn

from src.app.api import app as api_app


# Expose the FastAPI application at module level
app = api_app


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))The `vercel.json` schema validation failed with the following message: should NOT have additional property `excludeFiles`
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

