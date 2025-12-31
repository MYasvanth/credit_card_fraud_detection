@echo off
echo Starting BentoML Service on port 8080...
call C:\Users\admin\miniconda3\Scripts\activate.bat credit_card_fraud
echo Service will be available at: http://localhost:8080
python bento_flask_service.py
pause