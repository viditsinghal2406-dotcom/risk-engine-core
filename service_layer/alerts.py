# ============================================================
# Project 1a: Crypto Market Risk Intelligence System
# alerts.py -- Email and Slack alerting for critical anomalies
# ============================================================

import logging
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

from config import (
    ALERT_EMAIL_ENABLED, ALERT_EMAIL_FROM, ALERT_EMAIL_TO,
    ALERT_EMAIL_SMTP_URL, ALERT_EMAIL_PASSWORD,
    ALERT_SLACK_ENABLED, ALERT_SLACK_WEBHOOK,
    ALERT_RISK_THRESHOLD
)

logger = logging.getLogger(__name__)


def _safe_num(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


# ----------------------------------------------------------------
# EMAIL ALERTS
# ----------------------------------------------------------------

def send_email_alert(score: float, level: str, risk_data: dict) -> bool:
    """Send email alert for critical anomaly."""
    if not ALERT_EMAIL_ENABLED or not ALERT_EMAIL_SMTP_URL or not ALERT_EMAIL_PASSWORD:
        return False

    try:
        close_price = _safe_num(risk_data.get("close"), 0.0)
        if_score = _safe_num(risk_data.get("if_score"), 0.0)
        z_score = _safe_num(risk_data.get("z_score"), 0.0)
        lstm_score = _safe_num(risk_data.get("lstm_score"), 0.0)

        subject = f"🚨 CRITICAL: BTC Risk Score {score:.0f}/100 ({level})"
        
        body = f"""
        <html>
            <body style="font-family: Arial, sans-serif;">
                <h2 style="color: #d32f2f;">BTC Market Risk Alert</h2>
                <table style="border-collapse: collapse; width: 100%; margin: 20px 0;">
                    <tr style="background-color: #f5f5f5;">
                        <td style="padding: 10px; border: 1px solid #ddd;"><b>Risk Score</b></td>
                        <td style="padding: 10px; border: 1px solid #ddd;">{score:.1f}/100</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #ddd;"><b>Level</b></td>
                        <td style="padding: 10px; border: 1px solid #ddd;"><b>{level}</b></td>
                    </tr>
                    <tr style="background-color: #f5f5f5;">
                        <td style="padding: 10px; border: 1px solid #ddd;"><b>Current Price</b></td>
                        <td style="padding: 10px; border: 1px solid #ddd;">${close_price:,.2f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #ddd;"><b>Isolation Forest</b></td>
                        <td style="padding: 10px; border: 1px solid #ddd;">{if_score:.1f}</td>
                    </tr>
                    <tr style="background-color: #f5f5f5;">
                        <td style="padding: 10px; border: 1px solid #ddd;"><b>Z-Score</b></td>
                        <td style="padding: 10px; border: 1px solid #ddd;">{z_score:.1f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border: 1px solid #ddd;"><b>LSTM Neural Net</b></td>
                        <td style="padding: 10px; border: 1px solid #ddd;">{lstm_score:.1f}</td>
                    </tr>
                </table>
                <p style="color: #666; font-size: 12px;">Generated at {risk_data.get('timestamp', 'Unknown')}</p>
            </body>
        </html>
        """

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = ALERT_EMAIL_FROM
        msg["To"] = ALERT_EMAIL_TO
        msg.attach(MIMEText(body, "html"))

        host, port = ALERT_EMAIL_SMTP_URL.split(":")
        port = int(port)
        
        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(ALERT_EMAIL_FROM, ALERT_EMAIL_PASSWORD)
            server.sendmail(ALERT_EMAIL_FROM, ALERT_EMAIL_TO, msg.as_string())

        logger.info(f"Email alert sent to {ALERT_EMAIL_TO} for score {score:.1f}")
        return True

    except Exception as e:
        logger.error(f"Failed to send email alert: {e}")
        return False


# ----------------------------------------------------------------
# SLACK ALERTS
# ----------------------------------------------------------------

def send_slack_alert(score: float, level: str, risk_data: dict) -> bool:
    """Send Slack message for critical anomaly."""
    if not ALERT_SLACK_ENABLED or not ALERT_SLACK_WEBHOOK:
        return False

    try:
        close_price = _safe_num(risk_data.get("close"), 0.0)
        if_score = _safe_num(risk_data.get("if_score"), 0.0)
        z_score = _safe_num(risk_data.get("z_score"), 0.0)
        lstm_score = _safe_num(risk_data.get("lstm_score"), 0.0)

        color_map = {
            "Critical": "#d32f2f",
            "High": "#f57c00",
            "Medium": "#fbc02d",
            "Low": "#388e3c",
        }

        payload = {
            "attachments": [
                {
                    "color": color_map.get(level, "#9c27b0"),
                    "title": f"🚨 CRITICAL ALERT: Risk Score {score:.0f}/100",
                    "fields": [
                        {
                            "title": "Level",
                            "value": level,
                            "short": True
                        },
                        {
                            "title": "Current Price",
                            "value": f"${close_price:,.2f}",
                            "short": True
                        },
                        {
                            "title": "Isolation Forest",
                            "value": f"{if_score:.1f}",
                            "short": True
                        },
                        {
                            "title": "Z-Score",
                            "value": f"{z_score:.1f}",
                            "short": True
                        },
                        {
                            "title": "LSTM Neural Net",
                            "value": f"{lstm_score:.1f}",
                            "short": True
                        },
                        {
                            "title": "Timestamp",
                            "value": risk_data.get('timestamp', 'Unknown'),
                            "short": True
                        },
                        {
                            "title": "Summary",
                            "value": risk_data.get('summary', 'Anomaly detected in market data'),
                            "short": False
                        }
                    ]
                }
            ]
        }

        response = requests.post(ALERT_SLACK_WEBHOOK, json=payload, timeout=5)
        response.raise_for_status()
        
        logger.info(f"Slack alert sent for score {score:.1f}")
        return True

    except Exception as e:
        logger.error(f"Failed to send Slack alert: {e}")
        return False


# ----------------------------------------------------------------
# UNIFIED ALERT DISPATCH
# ----------------------------------------------------------------

def send_critical_alert(score: float, level: str, risk_data: dict) -> int:
    """
    Send alerts via all enabled channels.
    Returns count of successfully sent alerts.
    """
    if score < ALERT_RISK_THRESHOLD:
        return 0

    sent_count = 0
    
    if ALERT_EMAIL_ENABLED:
        if send_email_alert(score, level, risk_data):
            sent_count += 1
    
    if ALERT_SLACK_ENABLED:
        if send_slack_alert(score, level, risk_data):
            sent_count += 1

    return sent_count
