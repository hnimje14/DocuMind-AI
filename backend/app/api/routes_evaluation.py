from typing import Optional
from fastapi import APIRouter, Depends
from backend.app.api.dependencies import ServiceContainer, get_container
from backend.app.schemas.models import (
    EvaluationRunRequest,
    EvaluationRunResponse,
)

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


@router.post("/run", response_model=EvaluationRunResponse)
def run_evaluation_pipeline(
    request: EvaluationRunRequest = EvaluationRunRequest(),
    container: ServiceContainer = Depends(get_container),
) -> EvaluationRunResponse:
    """Run the reproducible retrieval and generation evaluation benchmark."""
    return container.evaluator.run_evaluation(request)


@router.get("/latest", response_model=Optional[EvaluationRunResponse])
def get_latest_evaluation_results(
    container: ServiceContainer = Depends(get_container),
) -> Optional[EvaluationRunResponse]:
    """Retrieve the most recent evaluation run results if available."""
    return container.evaluator.get_latest_evaluation()
