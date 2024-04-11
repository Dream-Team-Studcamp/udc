FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app

ENV PORT=8080