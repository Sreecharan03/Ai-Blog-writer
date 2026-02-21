# app/api/auth.py
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from app.core.config import _load_env_once

router = APIRouter(tags=["auth"])
_bearer = HTTPBearer(auto_error=False)


# ----------------------------
# Minimal HS256 JWT (no extra libs)
# ----------------------------
def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def _b64url_decode(data: str) -> bytes:
    pad = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + pad).encode("utf-8"))


def _jwt_secret() -> str:
    _load_env_once()
    secret = os.getenv("JWT_SECRET_KEY", "").strip()
    if not secret or secret == "CHANGE_ME_LONG_RANDOM":
        raise RuntimeError("JWT_SECRET_KEY is missing/weak. Set a strong value in .env")
    return secret


def jwt_encode(payload: Dict[str, Any], secret: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")

    sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    sig_b64 = _b64url_encode(sig)
    return f"{header_b64}.{payload_b64}.{sig_b64}"


def jwt_decode(token: str, secret: str) -> Dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3:
        raise HTTPException(status_code=401, detail="Invalid token format")

    header_b64, payload_b64, sig_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")

    expected = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    got = _b64url_decode(sig_b64)

    if not hmac.compare_digest(expected, got):
        raise HTTPException(status_code=401, detail="Invalid token signature")

    payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))

    now = int(time.time())
    exp = payload.get("exp")
    if exp is not None and now >= int(exp):
        raise HTTPException(status_code=401, detail="Token expired")

    return payload


# ----------------------------
# API models
# ----------------------------
class LoginRequest(BaseModel):
    tenant_id: str = Field(..., description="Tenant UUID")
    user_id: Optional[str] = Field(None, description="Optional user UUID. If omitted, server generates one.")
    role: str = Field("tenant_admin", description="Role for this token (dev flow).")


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    tenant_id: str
    user_id: str
    role: str


# ----------------------------
# Dependencies: tenant context + role checks
# ----------------------------
def get_current_claims(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Dict[str, Any]:
    if not creds or creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    secret = _jwt_secret()
    return jwt_decode(creds.credentials, secret)


def require_roles(*allowed_roles: str):
    def _dep(claims: Dict[str, Any] = Depends(get_current_claims)) -> Dict[str, Any]:
        role = claims.get("role")
        if allowed_roles and role not in allowed_roles:
            raise HTTPException(status_code=403, detail="Forbidden (role)")
        return claims

    return _dep


# ----------------------------
# Endpoints
# ----------------------------
@router.post("/auth/login", response_model=LoginResponse)
def login(body: LoginRequest) -> LoginResponse:
    """
    Dev-friendly login: issues a JWT for a tenant_id + role.
    (Later we will replace with real user auth.)
    """
    now = int(time.time())
    exp_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "720"))
    exp = now + (exp_minutes * 60)

    user_id = body.user_id or str(uuid.uuid4())

    claims = {
        "sub": user_id,
        "tenant_id": body.tenant_id,
        "role": body.role,
        "iat": now,
        "exp": exp,
    }

    token = jwt_encode(claims, _jwt_secret())
    return LoginResponse(
        access_token=token,
        expires_in=exp - now,
        tenant_id=body.tenant_id,
        user_id=user_id,
        role=body.role,
    )


@router.post("/auth/logout")
def logout():
    """
    JWT logout is optional (blacklist). For now just return ok.
    """
    return {"status": "ok"}


@router.get("/me")
def me(claims: Dict[str, Any] = Depends(get_current_claims)):
    """
    Return current user profile + tenant context (from token).
    """
    return {
        "status": "ok",
        "user": {"user_id": claims.get("sub"), "role": claims.get("role")},
        "tenant": {"tenant_id": claims.get("tenant_id")},
        "claims": claims,
    }


# Role-check demo endpoint (to prove Day-4 role checks)
@router.get("/admin/ping")
def admin_ping(claims: Dict[str, Any] = Depends(require_roles("tenant_admin"))):
    return {"status": "ok", "msg": "admin access granted", "tenant_id": claims.get("tenant_id")}
