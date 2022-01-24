FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./eeg.ini /code/eeg.ini
COPY ./cmuh.ini /code/cmuh.ini
COPY ./t0.eeg /code/t0.eeg
COPY ./Best_model /code/Best_model

RUN apt-get update && apt-get install ffmpeg -y
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

CMD python -m uvicorn app.main:app --host 0.0.0.0 --port 80 --reload
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]