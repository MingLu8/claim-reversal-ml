from fastapi import FastAPI
app = FastAPI()

from src.routes.home import router as home_router
app.include_router(home_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)