@echo off
echo Starting Expense Tracker Application...
echo.
echo 1. Starting MongoDB (make sure MongoDB is running)
echo 2. Starting Backend Server...
echo 3. Starting Frontend Application...
echo.
echo Please wait...
echo.

REM Start the backend server
start "Backend Server" cmd /k "npm run server"

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul

REM Start the frontend
start "Frontend App" cmd /k "npm start"

echo.
echo Application is starting up!
echo Backend will be available at: http://localhost:5000
echo Frontend will be available at: http://localhost:3000
echo.
echo Press any key to close this window...
pause >nul


