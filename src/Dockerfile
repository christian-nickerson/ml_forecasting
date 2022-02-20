FROM python:3.9-slim

COPY /src/ /stock/src/
COPY requirements.txt /stock/

WORKDIR /stock/

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "-u", "src/main.py"]
