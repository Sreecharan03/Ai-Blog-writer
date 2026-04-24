# app/api/auth.py
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import smtplib
import time
import uuid
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

import bcrypt as _bcrypt_lib
import psycopg2
import psycopg2.extras
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field

from app.core.config import _load_env_once

router = APIRouter(tags=["auth"])
_bearer = HTTPBearer(auto_error=False)


def _hash_password(password: str) -> str:
    """SHA-256 digest (32 bytes) → bcrypt. Bypasses the 72-byte bcrypt limit."""
    digest = hashlib.sha256(password.encode()).digest()
    return _bcrypt_lib.hashpw(digest, _bcrypt_lib.gensalt(rounds=12)).decode()


def _verify_password(password: str, hashed: str) -> bool:
    digest = hashlib.sha256(password.encode()).digest()
    return _bcrypt_lib.checkpw(digest, hashed.encode())


# ============================================================
# Helpers — env / DB
# ============================================================
def _env(key: str, default: str = "") -> str:
    _load_env_once()
    return os.getenv(key, default).strip()


def _require_env(key: str) -> str:
    val = _env(key)
    if not val:
        raise RuntimeError(f"Missing required env var: {key}")
    return val


def _db():
    _load_env_once()
    return psycopg2.connect(
        host=_require_env("DB_HOST"),
        port=int(_env("DB_PORT", "5432")),
        dbname=_require_env("DB_NAME"),
        user=_require_env("DB_USER"),
        password=_require_env("DB_PASSWORD"),
        sslmode=_env("DB_SSLMODE", "require"),
        connect_timeout=8,
    )


# ============================================================
# JWT (minimal HS256 — no extra libs)
# ============================================================
def _jwt_secret() -> str:
    secret = _env("JWT_SECRET_KEY")
    if not secret or secret == "CHANGE_ME_LONG_RANDOM":
        raise RuntimeError("JWT_SECRET_KEY is missing or weak. Set a strong value in .env")
    return secret


def _b64url_enc(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_dec(data: str) -> bytes:
    pad = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + pad).encode())


def jwt_encode(payload: Dict[str, Any], secret: str) -> str:
    header = _b64url_enc(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    body = _b64url_enc(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
    sig = _b64url_enc(hmac.new(secret.encode(), f"{header}.{body}".encode(), hashlib.sha256).digest())
    return f"{header}.{body}.{sig}"


def jwt_decode(token: str, secret: str) -> Dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3:
        raise HTTPException(status_code=401, detail="Invalid token format")
    header, body, sig = parts
    expected = _b64url_enc(hmac.new(secret.encode(), f"{header}.{body}".encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(expected, sig):
        raise HTTPException(status_code=401, detail="Invalid token signature")
    payload = json.loads(_b64url_dec(body))
    if payload.get("exp") and int(time.time()) >= int(payload["exp"]):
        raise HTTPException(status_code=401, detail="Token expired")
    return payload


def _make_token(user_id: str, tenant_id: str, role: str, jti: str) -> str:
    now = int(time.time())
    exp_min = int(_env("ACCESS_TOKEN_EXPIRE_MINUTES", "720"))
    return jwt_encode({
        "sub": user_id,
        "tenant_id": tenant_id,
        "role": role,
        "jti": jti,
        "iat": now,
        "exp": now + exp_min * 60,
    }, _jwt_secret())


# ============================================================
# Token helpers (password reset / email verify)
# ============================================================
def _generate_otp() -> tuple[str, str]:
    """Returns (6-digit code, sha256_hex_hash). Store only the hash."""
    code = f"{secrets.randbelow(1_000_000):06d}"
    hashed = hashlib.sha256(code.encode()).hexdigest()
    return code, hashed


# ============================================================
# Email sender
# ============================================================
def _smtp_send(host: str, port: int, user: str, password: str, mail_from: str, to: str, raw: str) -> None:
    with smtplib.SMTP(host, port, timeout=15) as s:
        s.ehlo()
        s.starttls()
        s.login(user, password)
        s.sendmail(mail_from, [to], raw)


def _send_email(to: str, subject: str, html_body: str) -> None:
    # --- primary: Gmail ---
    primary = {
        "host": _require_env("SMTP_HOST"),
        "port": int(_env("SMTP_PORT", "587")),
        "user": _require_env("SMTP_USER"),
        "password": _require_env("SMTP_PASS"),
        "mail_from": _env("MAIL_FROM", _require_env("SMTP_USER")),
    }
    # --- fallback: Brevo ---
    fallback_user = _env("SMTP_FALLBACK_USER", "")
    fallback = {
        "host": _env("SMTP_FALLBACK_HOST", ""),
        "port": int(_env("SMTP_FALLBACK_PORT", "587")),
        "user": fallback_user,
        "password": _env("SMTP_FALLBACK_PASS", ""),
        "mail_from": _env("SMTP_FALLBACK_FROM", fallback_user),
    } if fallback_user else None

    for cfg in ([primary] + ([fallback] if fallback else [])):
        # Build a fresh message per relay so From header is always correct
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = cfg["mail_from"]
        msg["To"] = to
        msg.attach(MIMEText(html_body, "html"))
        try:
            _smtp_send(cfg["host"], cfg["port"], cfg["user"], cfg["password"], cfg["mail_from"], to, msg.as_string())
            logger.info("Email sent via %s to %s", cfg["host"], to)
            return
        except Exception as exc:
            logger.warning("SMTP via %s failed: %s — trying next relay", cfg["host"], exc)

    raise RuntimeError("All SMTP relays failed — email not delivered")


def _send_reset_email(to: str, code: str) -> None:
    _send_email(
        to=to,
        subject="Your Sighnal password reset code",
        html_body=f"""
        <p>You requested a password reset for your Sighnal account.</p>
        <p>Your reset code is:</p>
        <h2 style="letter-spacing:8px;font-size:36px;">{code}</h2>
        <p>Enter this code along with your new password to complete the reset.</p>
        <p>Expires in <strong>30 minutes</strong>. Single use only.</p>
        <p>If you did not request this, ignore this email — your account is safe.</p>
        """,
    )


def _send_verify_email(to: str, code: str) -> None:
    _send_email(
        to=to,
        subject="Your Sighnal verification code",
        html_body=f"""
        <p>Welcome to Sighnal! Use the code below to verify your email address.</p>
        <h2 style="letter-spacing:8px;font-size:36px;">{code}</h2>
        <p>Enter this code to complete your registration.</p>
        <p>Expires in <strong>24 hours</strong>. Single use only.</p>
        """,
    )


# ============================================================
# Auth dependency — used by all protected endpoints
# ============================================================
def get_current_claims(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Dict[str, Any]:
    if not creds or creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    claims = jwt_decode(creds.credentials, _jwt_secret())

    # Check session not revoked
    jti = claims.get("jti")
    if jti:
        try:
            with _db() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT revoked_at FROM public.user_sessions WHERE jti=%s LIMIT 1",
                        (jti,),
                    )
                    row = cur.fetchone()
                    if row and row[0] is not None:
                        raise HTTPException(status_code=401, detail="Session has been revoked")
        except HTTPException:
            raise
        except Exception:
            pass  # If session table unavailable, allow (graceful degradation)

    return claims


def require_roles(*allowed_roles: str):
    def _dep(claims: Dict[str, Any] = Depends(get_current_claims)) -> Dict[str, Any]:
        if allowed_roles and claims.get("role") not in allowed_roles:
            raise HTTPException(status_code=403, detail="Forbidden")
        return claims
    return _dep


# ============================================================
# Request / Response models
# ============================================================
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, description="Minimum 8 characters")
    full_name: Optional[str] = None
    tenant_name: Optional[str] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class VerifyEmailRequest(BaseModel):
    email: EmailStr
    code: str = Field(..., min_length=6, max_length=6, pattern=r"^\d{6}$")


class ResetPasswordRequest(BaseModel):
    email: EmailStr
    code: str = Field(..., min_length=6, max_length=6, pattern=r"^\d{6}$")
    new_password: str = Field(..., min_length=8)


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str
    tenant_id: str
    role: str
    email: str


# ============================================================
# POST /auth/register
# ============================================================
@router.post("/auth/register", response_model=TokenResponse, status_code=201)
def register(body: RegisterRequest):
    """
    Create a new tenant + user. Sends a verification email.
    Returns a JWT so the user is immediately logged in.
    """
    pw_hash = _hash_password(body.password)
    jti = str(uuid.uuid4())
    tenant_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    exp_min = int(_env("ACCESS_TOKEN_EXPIRE_MINUTES", "720"))

    raw_verify, hash_verify = _generate_otp()

    try:
        with _db() as conn:
            with conn.cursor() as cur:
                # 1. Create tenant (both tables — tenants_fin is required by job_events FK)
                tenant_name = body.tenant_name or body.full_name or body.email.split("@")[0]
                cur.execute(
                    "INSERT INTO public.tenants (tenant_id, name) VALUES (%s, %s)",
                    (tenant_id, tenant_name),
                )
                cur.execute(
                    "INSERT INTO public.tenants_fin (tenant_id, name) VALUES (%s, %s) ON CONFLICT (tenant_id) DO NOTHING",
                    (tenant_id, tenant_name),
                )

                # 2. Create user
                try:
                    cur.execute(
                        """
                        INSERT INTO public.users
                            (user_id, tenant_id, email, password_hash, full_name, role)
                        VALUES (%s, %s, %s, %s, %s, 'tenant_admin')
                        """,
                        (user_id, tenant_id, body.email, pw_hash, body.full_name),
                    )
                except psycopg2.errors.UniqueViolation:
                    raise HTTPException(status_code=409, detail="Email already registered")

                # 3. Email verification token
                cur.execute(
                    """
                    INSERT INTO public.email_verification_tokens (user_id, token_hash)
                    VALUES (%s, %s)
                    """,
                    (user_id, hash_verify),
                )

                # 4. Session
                cur.execute(
                    """
                    INSERT INTO public.user_sessions
                        (user_id, tenant_id, jti, created_at, expires_at)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (user_id, tenant_id, jti, now,
                     datetime.fromtimestamp(time.time() + exp_min * 60, tz=timezone.utc)),
                )
            conn.commit()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)[:200]}")

    # Send verification email (non-blocking best-effort)
    try:
        _send_verify_email(body.email, raw_verify)
    except Exception:
        pass  # Don't fail registration if email send fails

    token = _make_token(user_id, tenant_id, "tenant_admin", jti)
    return TokenResponse(
        access_token=token,
        expires_in=exp_min * 60,
        user_id=user_id,
        tenant_id=tenant_id,
        role="tenant_admin",
        email=body.email,
    )


# ============================================================
# POST /auth/login
# ============================================================
@router.post("/auth/login", response_model=TokenResponse)
def login(body: LoginRequest):
    """
    Email + password login. Returns JWT.
    Locks account for 15 minutes after 5 consecutive failures.
    """
    try:
        with _db() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT user_id, tenant_id, email, password_hash, role,
                           is_active, failed_login_count, locked_until
                    FROM public.users WHERE email=%s LIMIT 1
                    """,
                    (body.email,),
                )
                user = cur.fetchone()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {str(e)[:200]}")

    # Always run hash verify to prevent timing attacks (even if user not found)
    dummy_hash = "$2b$12$KIXFjH9gvMoXqxhXmkUXuuSmNvdqSzBt4G8.0zJgI6eWzFq3lK6i6"
    candidate_hash = user["password_hash"] if user else dummy_hash

    if not _verify_password(body.password, candidate_hash) or not user:
        if user:
            # Increment failed login count and possibly lock
            with _db() as conn:
                with conn.cursor() as cur:
                    new_count = int(user["failed_login_count"] or 0) + 1
                    locked_until = None
                    if new_count >= 5:
                        locked_until = datetime.fromtimestamp(time.time() + 900, tz=timezone.utc)
                    cur.execute(
                        "UPDATE public.users SET failed_login_count=%s, locked_until=%s WHERE user_id=%s",
                        (new_count, locked_until, str(user["user_id"])),
                    )
                conn.commit()
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not user["is_active"]:
        raise HTTPException(status_code=403, detail="Account is disabled")

    if user["locked_until"] and user["locked_until"] > datetime.now(timezone.utc):
        raise HTTPException(status_code=429, detail="Account locked due to too many failed attempts. Try again in 15 minutes.")

    jti = str(uuid.uuid4())
    exp_min = int(_env("ACCESS_TOKEN_EXPIRE_MINUTES", "720"))
    now = datetime.now(timezone.utc)

    try:
        with _db() as conn:
            with conn.cursor() as cur:
                # Reset failed count + update last login
                cur.execute(
                    "UPDATE public.users SET failed_login_count=0, locked_until=NULL, last_login_at=%s WHERE user_id=%s",
                    (now, str(user["user_id"])),
                )
                # Create session
                cur.execute(
                    """
                    INSERT INTO public.user_sessions
                        (user_id, tenant_id, jti, created_at, expires_at)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (str(user["user_id"]), str(user["tenant_id"]), jti, now,
                     datetime.fromtimestamp(time.time() + exp_min * 60, tz=timezone.utc)),
                )
            conn.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session error: {str(e)[:200]}")

    token = _make_token(str(user["user_id"]), str(user["tenant_id"]), user["role"], jti)
    return TokenResponse(
        access_token=token,
        expires_in=exp_min * 60,
        user_id=str(user["user_id"]),
        tenant_id=str(user["tenant_id"]),
        role=user["role"],
        email=user["email"],
    )


# ============================================================
# POST /auth/logout
# ============================================================
@router.post("/auth/logout")
def logout(claims: Dict[str, Any] = Depends(get_current_claims)):
    """Revoke the current session. Token becomes invalid immediately."""
    jti = claims.get("jti")
    if jti:
        try:
            with _db() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE public.user_sessions SET revoked_at=%s WHERE jti=%s",
                        (datetime.now(timezone.utc), jti),
                    )
                conn.commit()
        except Exception:
            pass
    return {"status": "ok", "message": "Logged out successfully"}


# ============================================================
# POST /auth/forgot-password
# ============================================================
@router.post("/auth/forgot-password")
def forgot_password(body: ForgotPasswordRequest):
    """
    Send a password reset link to the user's email.
    Always returns 200 — never leaks whether email exists.
    """
    raw_token, token_hash = _generate_otp()

    try:
        with _db() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT user_id FROM public.users WHERE email=%s AND is_active=TRUE LIMIT 1",
                    (body.email,),
                )
                user = cur.fetchone()

            if user:
                with conn.cursor() as cur:
                    # Invalidate any existing unused tokens for this user
                    cur.execute(
                        "UPDATE public.password_reset_tokens SET used_at=now() WHERE user_id=%s AND used_at IS NULL",
                        (str(user["user_id"]),),
                    )
                    # Insert new token
                    cur.execute(
                        "INSERT INTO public.password_reset_tokens (user_id, token_hash) VALUES (%s, %s)",
                        (str(user["user_id"]), token_hash),
                    )
                conn.commit()
    except Exception:
        pass  # Silently fail — don't expose DB errors

    # Send email if user exists (best-effort, non-blocking)
    try:
        with _db() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT user_id FROM public.users WHERE email=%s LIMIT 1", (body.email,))
                if cur.fetchone():
                    _send_reset_email(body.email, raw_token)
    except Exception as exc:
        logger.error("forgot-password email delivery failed for %s: %s", body.email, exc)

    return {"status": "ok", "message": "If that email is registered, a 6-digit reset code has been sent."}


# ============================================================
# POST /auth/reset-password
# ============================================================
@router.post("/auth/reset-password")
def reset_password(body: ResetPasswordRequest):
    """
    Verify reset token and set a new password.
    Token is single-use and expires in 30 minutes.
    """
    code_hash = hashlib.sha256(body.code.encode()).hexdigest()

    try:
        with _db() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT prt.token_id, prt.user_id, prt.expires_at, prt.used_at
                    FROM public.password_reset_tokens prt
                    JOIN public.users u ON u.user_id = prt.user_id
                    WHERE u.email = %s AND prt.token_hash = %s
                    LIMIT 1
                    """,
                    (body.email, code_hash),
                )
                row = cur.fetchone()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {str(e)[:200]}")

    if not row:
        raise HTTPException(status_code=400, detail="Invalid or expired code")
    if row["used_at"] is not None:
        raise HTTPException(status_code=400, detail="Code has already been used")
    if row["expires_at"] < datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="Code has expired. Request a new one.")

    new_hash = _hash_password(body.new_password)

    try:
        with _db() as conn:
            with conn.cursor() as cur:
                # Update password
                cur.execute(
                    "UPDATE public.users SET password_hash=%s, failed_login_count=0, locked_until=NULL WHERE user_id=%s",
                    (new_hash, str(row["user_id"])),
                )
                # Mark token used
                cur.execute(
                    "UPDATE public.password_reset_tokens SET used_at=now() WHERE token_id=%s",
                    (str(row["token_id"]),),
                )
                # Revoke all active sessions (force re-login with new password)
                cur.execute(
                    "UPDATE public.user_sessions SET revoked_at=now() WHERE user_id=%s AND revoked_at IS NULL",
                    (str(row["user_id"]),),
                )
            conn.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)[:200]}")

    return {"status": "ok", "message": "Password reset successful. Please log in with your new password."}


# ============================================================
# GET /auth/verify-email?token=...
# ============================================================
@router.post("/auth/verify-email")
def verify_email(body: VerifyEmailRequest):
    """Verify email address using the 6-digit code sent after registration."""
    code_hash = hashlib.sha256(body.code.encode()).hexdigest()

    try:
        with _db() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT evt.token_id, evt.user_id, evt.expires_at, evt.used_at
                    FROM public.email_verification_tokens evt
                    JOIN public.users u ON u.user_id = evt.user_id
                    WHERE u.email = %s AND evt.token_hash = %s
                    LIMIT 1
                    """,
                    (body.email, code_hash),
                )
                row = cur.fetchone()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {str(e)[:200]}")

    if not row:
        raise HTTPException(status_code=400, detail="Invalid verification code")
    if row["used_at"] is not None:
        return {"status": "ok", "message": "Email already verified"}
    if row["expires_at"] < datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="Verification code expired. Please register again.")

    try:
        with _db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE public.users SET is_email_verified=TRUE WHERE user_id=%s",
                    (str(row["user_id"]),),
                )
                cur.execute(
                    "UPDATE public.email_verification_tokens SET used_at=now() WHERE token_id=%s",
                    (str(row["token_id"]),),
                )
            conn.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)[:200]}")

    return {"status": "ok", "message": "Email verified successfully"}


# ============================================================
# POST /auth/resend-verification
# ============================================================
@router.post("/auth/resend-verification")
def resend_verification(body: ForgotPasswordRequest):
    """
    Send a fresh 6-digit verification code to an unverified account.
    Always returns 200 — never leaks whether email exists.
    """
    code, code_hash = _generate_otp()

    try:
        with _db() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT user_id, is_email_verified FROM public.users WHERE email=%s AND is_active=TRUE LIMIT 1",
                    (body.email,),
                )
                user = cur.fetchone()

            if user and not user["is_email_verified"]:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE public.email_verification_tokens SET used_at=now() WHERE user_id=%s AND used_at IS NULL",
                        (str(user["user_id"]),),
                    )
                    cur.execute(
                        "INSERT INTO public.email_verification_tokens (user_id, token_hash) VALUES (%s, %s)",
                        (str(user["user_id"]), code_hash),
                    )
                conn.commit()
                try:
                    _send_verify_email(body.email, code)
                except Exception as exc:
                    logger.error("resend-verification email failed for %s: %s", body.email, exc)
    except Exception as exc:
        logger.error("resend-verification DB error for %s: %s", body.email, exc)

    return {"status": "ok", "message": "If that email is registered and unverified, a new code has been sent."}


# ============================================================
# POST /auth/change-password  (requires login)
# ============================================================
@router.post("/auth/change-password")
def change_password(
    body: ChangePasswordRequest,
    claims: Dict[str, Any] = Depends(get_current_claims),
):
    """Change password while logged in. Requires current password."""
    user_id = claims.get("sub")

    try:
        with _db() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT password_hash FROM public.users WHERE user_id=%s LIMIT 1",
                    (user_id,),
                )
                user = cur.fetchone()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {str(e)[:200]}")

    if not user or not _verify_password(body.current_password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Current password is incorrect")

    new_hash = _hash_password(body.new_password)
    try:
        with _db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE public.users SET password_hash=%s WHERE user_id=%s",
                    (new_hash, user_id),
                )
            conn.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Change failed: {str(e)[:200]}")

    return {"status": "ok", "message": "Password changed successfully"}


# ============================================================
# GET /auth/me
# ============================================================
@router.get("/auth/me")
def me(claims: Dict[str, Any] = Depends(get_current_claims)):
    """Return current user profile."""
    user_id = claims.get("sub")
    try:
        with _db() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT user_id, tenant_id, email, full_name, role, is_email_verified, last_login_at, created_at FROM public.users WHERE user_id=%s LIMIT 1",
                    (user_id,),
                )
                user = cur.fetchone()
    except Exception:
        user = None

    return {
        "status": "ok",
        "user": {
            "user_id": claims.get("sub"),
            "tenant_id": claims.get("tenant_id"),
            "role": claims.get("role"),
            "email": user["email"] if user else None,
            "full_name": user["full_name"] if user else None,
            "is_email_verified": user["is_email_verified"] if user else None,
            "last_login_at": user["last_login_at"].isoformat() if user and user["last_login_at"] else None,
        },
    }


# ============================================================
# GET /admin/ping  (role check demo)
# ============================================================
@router.get("/admin/ping")
def admin_ping(claims: Dict[str, Any] = Depends(require_roles("tenant_admin", "super_admin"))):
    return {"status": "ok", "msg": "admin access granted", "tenant_id": claims.get("tenant_id")}
