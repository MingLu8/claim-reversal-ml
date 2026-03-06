from fastapi import APIRouter
from fastapi_class import View

router = APIRouter()


@View(router)
class EventScanner:

    @router.get("/")
    def read_root():
        return {"Hello": "World"}