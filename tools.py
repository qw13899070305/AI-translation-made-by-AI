# tools.py
def get_weather(city):
    # 调用天气 API
    return f"{city}今天晴，20-28℃"

def send_email(to, subject, body):
    # 发送邮件
    return "邮件已发送"

TOOLS = {
    "get_weather": get_weather,
    "send_email": send_email,
}