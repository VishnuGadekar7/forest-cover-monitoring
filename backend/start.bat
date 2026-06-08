#!/bin/sh
. .venv\Scripts\activate
uvicorn app.main:app --port 8000 --timeout-keep-alive 600