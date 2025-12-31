@echo off
echo Starting BentoML Service...
call C:\Users\admin\miniconda3\Scripts\activate.bat credit_card_fraud
python bento_flask_service.py