# The load run's fixed-delay provider. Pinned to the versions `uv.lock` resolves
# for the gateway, so the mock runs the same web stack it is standing behind.
# Nothing of the gateway's own package is installed or copied here.
FROM python:3.12-slim

RUN pip install --no-cache-dir fastapi==0.141.1 uvicorn==0.52.4

WORKDIR /app
COPY main.py ./

USER 1000
EXPOSE 8090

CMD ["python", "main.py"]
