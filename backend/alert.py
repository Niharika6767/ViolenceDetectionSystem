import smtplib
from email.mime.text import MIMEText

def send_alert():
    sender_email = "niharikagoud45@gmail.com"
    receiver_email = "gniharika6767@gmail.com"
    password = "pbwr ktdl anwd bupz"

    subject = "Violence Detected Alert!"
    body = "Violence has been detected in the live stream. Immediate attention required."

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print("Alert email sent successfully.")
    except Exception as e:
        print("Error sending email:", e)
