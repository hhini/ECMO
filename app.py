import os
import sys

# Entry point redirecting execution to standalone web_app/app.py
APP_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_APP_PATH = os.path.join(APP_DIR, "web_app", "app.py")

with open(WEB_APP_PATH, "r", encoding="utf-8") as f:
    code = f.read()

exec(compile(code, WEB_APP_PATH, 'exec'))
