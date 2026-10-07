from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Reading(BaseModel):
    machine_id: str = Field(examples=["M001"], min_length=1)
    timestamp: datetime | None = Field(default=None, description="Defaults to one hour after the last stored observation.")
    voltage: float = Field(ge=0, le=400)
    rotation: float = Field(ge=0, le=1500)
    pressure: float = Field(ge=0, le=300)
    vibration: float = Field(ge=0, le=200)


class Factor(BaseModel):
    feature: str
    label: str
    value: float
    contribution: float


class Prediction(BaseModel):
    machine_id: str
    timestamp: datetime
    failure_probability: float
    risk_level: str
    model_version: str
    threshold: float
    top_factors: list[Factor]


class MachineRisk(BaseModel):
    machine_id: str
    timestamp: datetime
    failure_probability: float
    risk_level: str
    top_factor: str = ""


class Health(BaseModel):
    status: str
    model_version: str | None
    model_name: str | None
    model_loaded_at: str | None
    data_last_timestamp: datetime | None
    machines: int
