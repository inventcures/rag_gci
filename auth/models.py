from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class DeviceRegistration:
    user_id: str
    name: str
    phone_hash: str
    role: str
    language: str
    site_id: str
    abha_id: Optional[str] = None
    digital_literacy_score: Optional[int] = None
    registered_at: float = field(default_factory=time.time)


@dataclass
class AuthenticatedUser:
    user_id: str
    role: str
    site_id: str
