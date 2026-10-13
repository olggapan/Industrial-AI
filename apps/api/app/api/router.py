from fastapi import APIRouter

from app.api.routes import (
    agent,
    documents,
    evaluation,
    events,
    health,
    incidents,
    machines,
    rag,
    diagnosis,
)

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(machines.router, prefix="/machines", tags=["machines"])
api_router.include_router(incidents.router, prefix="/incidents", tags=["incidents"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(events.router, prefix="/events", tags=["events"])
api_router.include_router(agent.router, prefix="/agent", tags=["agent"])
api_router.include_router(evaluation.router, prefix="/evaluation", tags=["evaluation"])
api_router.include_router(rag.router, prefix="/rag", tags=["rag"])
api_router.include_router(diagnosis.router, prefix="/diagnosis", tags=["diagnosis"])
