"""
Mobile API Module for Palli Sahayak Android App.

Provides mobile-optimized REST endpoints under /api/mobile/v1/.
All endpoints require JWT authentication.
Delegates to existing services -- no business logic duplication.
"""

from mobile_api.router import mobile_router

__all__ = ["mobile_router"]
