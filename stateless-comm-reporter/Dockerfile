FROM python:3.12

WORKDIR /app

COPY requirements.txt .
COPY *.json .
COPY .env .
COPY . .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
