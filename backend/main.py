"""FastAPI app: router registration and static frontend serving. No business
logic lives here."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.roads import router as roads_router
from api.terrain import router as terrain_router
from api.upload import router as upload_router

app = FastAPI(title="TOPO2STL")

app.include_router(roads_router)
app.include_router(terrain_router)
app.include_router(upload_router)

# Serve the built React frontend at / when it exists (the local "just run it"
# path). In development, run the Vite dev server instead — it proxies /api here.
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

if FRONTEND_DIST.is_dir():
    app.mount(
        "/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend"
    )
else:

    @app.get("/")
    def root():
        return JSONResponse(
            {
                "message": (
                    "Frontend build not found. Run `npm install && npm run build` in "
                    "frontend/, or use the Vite dev server (`npm run dev`, "
                    "http://localhost:5173) during development."
                )
            }
        )
