FROM python:3.10

LABEL org.opencontainers.image.source=https://github.com/zaRizk7/abide-demo
LABEL org.opencontainers.image.description="A simple container code to train an autism classifier with ABIDE dataset."

RUN apt update && apt install build-essential -y

COPY abide_demo /abide_demo

RUN pip install --upgrade pip

RUN pip install -r /abide_demo/requirements.txt

ENTRYPOINT ["python", "/abide_demo/main.py"]
