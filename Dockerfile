FROM python:3.10

RUN apt update && apt install build-essential -y

COPY abide_demo .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "main.py"]
