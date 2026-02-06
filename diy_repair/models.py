"""
Data Models for Home DIY Repair Q&A Generation
Pydantic models for validation and structure enforcement
"""

import json
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class DIYRepairQA(BaseModel):
    """
    Pydantic model for Home DIY Repair Q&A pairs
    Ensures structured output with validation
    """
    question: str = Field(..., min_length=10, max_length=500, description="The DIY repair question")
    answer: str = Field(..., min_length=20, max_length=2000, description="Detailed answer to the question")
    equipment_problem: str = Field(..., min_length=5, max_length=200, description="Equipment or problem being addressed")
    tools_required: List[str] = Field(..., min_items=1, max_items=15, description="List of tools needed")
    steps: List[str] = Field(..., min_items=2, max_items=20, description="Step-by-step instructions")
    safety_info: str = Field(..., min_length=10, max_length=500, description="Safety information and warnings")
    tips: str = Field(..., min_length=10, max_length=500, description="Additional tips and best practices")

    @validator('tools_required')
    def validate_tools(cls, v):
        """Validate that tools are not empty strings"""
        if not all(tool.strip() for tool in v):
            raise ValueError("Tools cannot be empty strings")
        return [tool.strip() for tool in v]

    @validator('steps')
    def validate_steps(cls, v):
        """Validate that steps are not empty and properly formatted"""
        if not all(step.strip() for step in v):
            raise ValueError("Steps cannot be empty strings")
        return [step.strip() for step in v]

    @validator('question', 'answer', 'equipment_problem', 'safety_info', 'tips')
    def validate_text_fields(cls, v):
        """Validate text fields are not just whitespace"""
        if not v.strip():
            raise ValueError("Text fields cannot be empty or just whitespace")
        return v.strip()


class GenerationResult(BaseModel):
    """
    Model for tracking generation results and metadata
    """
    trace_id: str = Field(..., description="Unique identifier for this generation")
    qa_pair: Optional[DIYRepairQA] = Field(None, description="Generated Q&A pair if valid")
    raw_response: str = Field(..., description="Raw response from the model")
    is_valid: bool = Field(..., description="Whether the response passed validation")
    validation_errors: List[str] = Field(default_factory=list, description="List of validation errors if any")
    generation_timestamp: str = Field(..., description="Timestamp of generation")


class ValidationSummary(BaseModel):
    """
    Summary of validation results across all generated samples
    """
    total_generated: int = Field(..., description="Total number of samples generated")
    valid_samples: int = Field(..., description="Number of valid samples")
    invalid_samples: int = Field(..., description="Number of invalid samples")
    validation_rate: float = Field(..., description="Percentage of valid samples")
    common_errors: List[str] = Field(default_factory=list, description="Most common validation errors")


def validate_json_structure(json_str: str) -> tuple[bool, Optional[DIYRepairQA], List[str]]:
    """
    Validate JSON string against DIYRepairQA model

    Args:
        json_str: JSON string to validate

    Returns:
        Tuple of (is_valid, parsed_model, error_messages)
    """
    errors = []

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON format: {str(e)}")
        return False, None, errors

    try:
        qa_pair = DIYRepairQA(**data)
        return True, qa_pair, []
    except Exception as e:
        errors.append(f"Validation error: {str(e)}")
        return False, None, errors
