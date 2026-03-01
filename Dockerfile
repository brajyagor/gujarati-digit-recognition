FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt && python download_model.py

EXPOSE 10000

CMD ["python", "app.py"]
