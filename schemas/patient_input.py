from typing import Any, Dict, List
from pydantic import BaseModel, Field


class MediaItem(BaseModel):
    file_data: str = Field(
        description=("Raw file reference such as file path, URL, or base64 string.")
    )

    description: str = Field(
        description=(
            "Short human-readable explanation of what this file represents. "
            "Examples: 'knee swelling photo', 'x-ray of left leg', "
            "'blood test report', 'discharge summary'."
        )
    )


class PatientInput(BaseModel):
    """Raw inputs received from the patient or system."""

    patient_info: Dict[str, Any] = Field(default_factory=dict)

    surgery_protocol: str | List[str] = ""

    symptoms: str | None = None

    medications: str | List[str] | None = None

    images: List[MediaItem] | None = None

    documents: List[MediaItem] | None = None
