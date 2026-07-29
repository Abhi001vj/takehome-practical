"""HTTP serving layer.

The ASGI target is `support_router.api.service:app`. Named `service` rather than `app` so
the submodule does not shadow the exported object.
"""

from .service import app, create_app

__all__ = ["app", "create_app"]
