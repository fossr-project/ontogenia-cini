# app/main.py
from fastapi import FastAPI
from app.routers import cq_validation

app = FastAPI(
    title="CQ Verification and Generation API",
    description="APIs for competency question validation and generation",
    version="1.0.0"
)

# Include routers with prefixes and tags
app.include_router(cq_validation.router, prefix="/validate", tags=["CQ Validation"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
