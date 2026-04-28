def get_weather(city):
    return f"{city}: Sunny, 20-28°C"

def send_email(to, subject, body):
    return "Email sent successfully"

TOOLS = {
    "get_weather": get_weather,
    "send_email": send_email,
}