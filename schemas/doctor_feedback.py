from pydantic import BaseModel
from schemas.recovery_plan import RecoveryPlan
from schemas.risk_level import RiskLevel

class DoctorFeedback(BaseModel):
    patient_id: str | None = None
    instructions: str | None = None
    adjusted_recovery_plan: RecoveryPlan | None = None
    follow_up_required: bool = True
    risk_override: RiskLevel | None = None
