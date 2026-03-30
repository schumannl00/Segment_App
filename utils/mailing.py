import os
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()

def send_mail(receiver_email : str, subject : str , body : str ):
    sender_email = os.getenv("SMTP_EMAIL")
    password = os.getenv("SMTP_PASSWORD")

    # Create a proper Email object
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = f"Zesbo Message Client <{sender_email}>" 
    msg['To'] = receiver_email

    try:
        with smtplib.SMTP(os.getenv("SMTP_SERVER"), int(os.getenv("SMTP_PORT"))) as server:
            server.starttls()
            server.login(sender_email, password)
            server.send_message(msg) # Use send_message, not sendmail
            print(f"Email sent to {receiver_email} and verified!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    send_mail(
        "Leonard.Schumann@medizin.uni-leipzig.de",
        "System Alert: Test Connection",
        "This is an automated test from the Zesbo internal client."
    )