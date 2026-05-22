@echo off
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8001') do (
    taskkill /F /PID %%a
    echo Killed PID %%a on port 8001
)
pause
