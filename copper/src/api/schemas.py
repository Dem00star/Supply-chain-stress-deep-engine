from pydantic import BaseModel
from typing import Dict, List

class ForecastResponse(BaseModel):
    status: str
    target_variable: str
    forecast_horizon_days: int
    predictions: Dict[str, float]
    confidence_intervals: Dict[str, List[float]]