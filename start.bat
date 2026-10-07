@echo off
cd /d "%~dp0"
if not exist .env copy .env.example .env
docker compose up --build
pause
