from pydantic import BaseModel, Field

class PatientFeedback(BaseModel):
    patient_id: str | None = None
    confirmed: bool = False
    notes_for_modification: str | None = None
    questions: list[str] = Field(default_factory=list)
    new_symptoms: str | None = None
