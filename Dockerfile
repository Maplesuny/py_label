FROM python:3.10

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./t0.eeg /code/t0.eeg

RUN apt-get update && apt-get install ffmpeg -y
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN pip install mne
RUN pip install -U efficientnet

COPY ./app /code/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]