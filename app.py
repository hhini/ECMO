# -*- coding: utf-8 -*-
"""
Root entry point for ECMO Thrombosis AI Web Application.
Seamlessly redirects execution to web_app/app.py while setting up workspace paths.
"""
import os
import sys

APP_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_APP_DIR = os.path.join(APP_DIR, "web_app")

if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
if WEB_APP_DIR not in sys.path:
    sys.path.insert(0, WEB_APP_DIR)

WEB_APP_PATH = os.path.join(WEB_APP_DIR, "app.py")

globals()["__file__"] = WEB_APP_PATH

with open(WEB_APP_PATH, "r", encoding="utf-8") as f:
    code = f.read()

exec(compile(code, WEB_APP_PATH, 'exec'), globals())
