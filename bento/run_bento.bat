@echo off
echo 🚀 Starting BentoML-style Fraud Detection Service...

call C:\Users\admin\miniconda3\Scripts\activate.bat credit_card_fraud

echo 📦 Starting enhanced service...
start /B python bento_flask_service.py

echo ⏳ Waiting for service to start...
timeout /t 3 /nobreak > nul

echo 🧪 Running tests...
python scripts\run_enhanced_bento.py

pause