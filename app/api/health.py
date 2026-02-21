from __future__ import annotations

from fastapi import APIRouter
from app.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    s = get_settings()
    return {"status": "ok", "service": "sighnal-backend", "env": s.environment}


@router.get("/ready")
def ready():
    """
    Day-1 readiness (REAL):
    - Confirms GCS object list permission
    - Confirms required prefixes exist by checking for .keep objects:
        raw/.keep, processed/.keep, url_snapshots/.keep, articles/.keep
    """
    s = get_settings()

    checks = {
        "gcs": {
            "ok": False,
            "detail": None,
        }
    }

    try:
        from google.cloud import storage

        client = storage.Client(project=s.project_id)

        required_keep_objects = [
            f"{s.gcs_prefix_raw}.keep",
            f"{s.gcs_prefix_processed}.keep",
            f"{s.gcs_prefix_url_snapshots}.keep",
            f"{s.gcs_prefix_articles}.keep",
        ]

        found = {}

        # REAL check: each object must be listable
        for obj_name in required_keep_objects:
            blobs = list(
                client.list_blobs(
                    s.gcs_bucket_name,
                    prefix=obj_name,
                    max_results=1,
                )
            )
            found[obj_name] = bool(blobs)

        all_ok = all(found.values())

        checks["gcs"]["ok"] = all_ok
        checks["gcs"]["detail"] = {
            "bucket": s.gcs_bucket_name,
            "tested_operation": "objects.list",
            "required_keep_objects": found,
        }

    except Exception as e:
        checks["gcs"]["detail"] = str(e)

    status = "ready" if checks["gcs"]["ok"] else "not_ready"
    return {"status": status, "checks": checks}
