FROM ultralytics/ultralytics:latest

WORKDIR /TBank_app

COPY . .

RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--reload", "--port", "8000"]
