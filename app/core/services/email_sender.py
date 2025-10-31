
import os
import smtplib
from email.message import EmailMessage
from app.logger import get_logger
from typing import Optional
from dotenv import load_dotenv
from datetime import datetime
import requests
import markdown2

load_dotenv()

logger = get_logger(__name__)


def _get_smtp_config():
    return {
        "host": os.getenv("SMTP_HOST"),
        "port": int(os.getenv("SMTP_PORT", "0")) if os.getenv("SMTP_PORT") else None,
        "user": os.getenv("SMTP_USER"),
        "password": os.getenv("SMTP_PASSWORD"),
        "use_tls": os.getenv("SMTP_USE_TLS", "true").lower() in ("1", "true", "yes"),
        "from_addr": os.getenv("EMAIL_FROM"),
        "to_addr": os.getenv("EMAIL_TO"),
    }


def _claim_email_sent(redis_client, session_id: str) -> bool:
    """Try to claim the session email send using Redis SETNX.

    Returns True if this caller acquired the right to send the email (first caller),
    False if someone else already did.
    """
    try:
        if not redis_client:
            return True
        key = f"session:{session_id}:closure_email_sent"
        # set with nx=True and small TTL to avoid permanent key
        ok = redis_client.set(key, "1", nx=True, ex=60 * 60 * 24)
        return bool(ok)
    except Exception as e:
        logger.warning(f"[email_sender] Redis claim failed, allowing send to proceed: {e}")
        return True


def send_closure_email(session_id: str, follow_up_manager=None, reason: Optional[str] = None, redis_client=None, to_addr: Optional[str] = None, browser: Optional[str] = None, ip: Optional[str] = None, session_created=None, session_last=None) -> bool:
    """Compose and send a closure email for a session.

    This function is tolerant: if SMTP config or recipient is missing, it logs and returns False.
    It uses a Redis SETNX claim (if provided) to ensure only the first caller actually sends the email.
    Always fetches the latest transcript from the DB to ensure completeness.
    """
    try:
        cfg = _get_smtp_config()
        if not cfg.get("host") or not cfg.get("to_addr"):
            logger.info("[email_sender] SMTP or recipient not configured, skipping email send")
            return False

        # Idempotency claim
        proceed = _claim_email_sent(redis_client, session_id)
        if not proceed:
            logger.info(f"[email_sender] Closure email for session {session_id} already sent by another worker; skipping.")
            return False

        # Lazy import DB helper to avoid circular imports at module import time
        try:
            from app.api.v1.helpers import get_db_conn
        except Exception:
            get_db_conn = None

        # Always fetch transcript and session metadata from DB (best-effort)
        rows = []
        transcript_lines = []
        user_region = "unknown"
        try:
            if get_db_conn:
                conn = get_db_conn()
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT role, content, created_at
                    FROM messages
                    WHERE session_id = %s
                    ORDER BY created_at ASC
                    """,
                    (str(session_id),),
                )
                rows = cur.fetchall()

                for r in rows:
                    role, content, created_at = r[0], r[1], r[2]
                    # Format timestamp to readable string
                    if hasattr(created_at, "isoformat"):
                        ts = created_at.strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        ts = str(created_at)
                    transcript_lines.append(f"[{ts}] {role.upper()}: {content}")

                # Session metadata
                cur.execute(
                    """
                    SELECT browser, ip, created_at, last_interaction_at
                    FROM sessions
                    WHERE session_id = %s
                    LIMIT 1
                    """,
                    (str(session_id),),
                )
                srow = cur.fetchone()

                if srow:
                    browser = browser or srow[0]
                    ip = ip or srow[1]
                    session_created = session_created or srow[2]
                    session_last = session_last or srow[3]
                    # Lookup user region by IP (best effort)
                    if ip and ip not in ("unknown", None, ""):
                        try:
                            resp = requests.get(f"https://ipapi.co/{ip}/json/", timeout=3)
                            if resp.status_code == 200:
                                data = resp.json()
                                country = data.get("country_name")
                                # Try multiple possible region keys for robustness
                                region = data.get("region") or data.get("region_name") or data.get("state_prov")
                                city = data.get("city")
                                user_region = ", ".join([v for v in [city, region, country] if v]) or "unknown"
                        except Exception as e:
                            logger.info(f"[email_sender] IP region lookup failed: {e}")

                cur.close()
                conn.close()

        except Exception as e:
            logger.debug(f"[email_sender] DB fetch skipped/failed for session {session_id}: {e}")

        # Always use EMAIL_TO from .env for now, ignore user table and to_addr argument
        recipient = cfg.get("to_addr")

        # Fallback: attempt to use in-memory follow_up_manager conversation history
        if not transcript_lines and follow_up_manager is not None:
            try:
                sh = follow_up_manager.get_conversation_history(session_id)
                for m in sh:
                    role = m.get("role")
                    content = m.get("content")
                    transcript_lines.append(f"[in-memory] {role.upper()}: {content}")
            except Exception:
                pass

        # Compute duration
        duration_str = "unknown"
        try:
            delta = None
            if session_created and session_last:
                delta = session_last - session_created
            else:
                # try using message timestamps
                if rows and len(rows) >= 2:
                    first = rows[0][2]
                    last = rows[-1][2]
                    delta = last - first
            if delta:
                seconds = int(delta.total_seconds())
                mins, secs = divmod(seconds, 60)
                hours, mins = divmod(mins, 60)
                if hours:
                    duration_str = f"{hours}h {mins}m {secs}s"
                elif mins:
                    duration_str = f"{mins}m {secs}s"
                else:
                    duration_str = f"{secs}s"
        except Exception:
            duration_str = "unknown"

        subject = f"Chat session summary: {session_id}"
        # Plain text body
        body = []
        body.append(f"Session ID: {session_id}")
        if reason:
            body.append(f"Reason: {reason}")
        body.append(f"IP: {ip or 'unknown'}")
        body.append(f"Region: {user_region}")
        body.append(f"Browser: {browser or 'unknown'}")
        body.append(f"Duration: {duration_str}")
        body.append("")
        body.append("Chat transcript:")
        body.extend(transcript_lines or ["(no transcript available)"])
        text = "\n".join(body)

        # Markdown formatting for transcript (for bold, highlights, etc.)
        transcript_md = "\n".join(transcript_lines or ["(no transcript available)"])
        transcript_html = markdown2.markdown(transcript_md, extras=["fenced-code-blocks", "tables", "strike", "break-on-newline"])

        html = f"""
        <html>
        <body style='font-family:Segoe UI,Arial,sans-serif;background:#f7f9fb;padding:0;margin:0;'>
          <div style='max-width:600px;margin:40px auto;background:#fff;border-radius:8px;box-shadow:0 2px 8px #0001;padding:32px;'>
            <h2 style='color:#2d6cdf;margin-top:0;'>Chat Session Summary</h2>
            <table style='width:100%;margin-bottom:24px;font-size:15px;'>
              <tr><td style='color:#888;'>Session ID:</td><td>{session_id}</td></tr>
              {f"<tr><td style='color:#888;'>Reason:</td><td>{reason}</td></tr>" if reason else ''}
              <tr><td style='color:#888;'>IP:</td><td>{ip or 'unknown'}</td></tr>
              <tr><td style='color:#888;'>Region:</td><td>{user_region}</td></tr>
              <tr><td style='color:#888;'>Browser:</td><td>{browser or 'unknown'}</td></tr>
              <tr><td style='color:#888;'>Duration:</td><td>{duration_str}</td></tr>
            </table>
            <div style='margin-bottom:12px;font-weight:bold;color:#2d6cdf;'>Chat Transcript:</div>
            <div style='background:#f3f6fa;border-radius:6px;padding:16px 12px;font-size:14px;line-height:1.7;'>
              {transcript_html}
            </div>
            <div style='margin-top:32px;color:#aaa;font-size:13px;text-align:center;'>
              This is an automated summary from <b>DITS Chatbot</b>.
            </div>
          </div>
        </body>
        </html>
        """

        # Build email
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = cfg.get("from_addr") or cfg.get("user")
        msg["To"] = recipient or cfg.get("to_addr")
        msg.set_content(text)
        msg.add_alternative(html, subtype="html")

        # Send via SMTP
        host = cfg.get("host")
        port = cfg.get("port") or 25
        user = cfg.get("user")
        password = cfg.get("password")
        use_tls = cfg.get("use_tls", True)

        host = str(host)
        port = int(port) if port else 25
        if use_tls and port == 465:
            smtp = smtplib.SMTP_SSL(host, port, timeout=10)
        else:
            smtp = smtplib.SMTP(host, port, timeout=10)
            if use_tls:
                smtp.starttls()

        if user and password:
            smtp.login(user, password)
        smtp.send_message(msg)
        smtp.quit()
        logger.info(f"[email_sender] Sent session summary email for session {session_id} to {cfg.get('to_addr')}")
        return True
    except Exception as e:
        logger.exception(f"[email_sender] Failed to send session summary email for session {session_id}: {e}")
        return False
