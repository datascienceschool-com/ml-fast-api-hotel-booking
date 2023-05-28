FROM python:3.9.13

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app
COPY . /app

RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

CMD ["uvicorn", "app.main:api", "--host", "0.0.0.0", "--port", "80"]
HEALTHCHECK --interval=180s --timeout=60s --start-period=25s --retries=3 CMD curl -f https://localhost:80/health


# RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
# USER appuser

# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
# HEALTHCHECK --interval=180s --timeout=60s --start-period=25s --retries=3 CMD curl -f https://localhost:80/health