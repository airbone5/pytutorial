FROM python:3.10

#WORKDIR /code

COPY ./requirements.txt /tmp/requirements.txt

RUN apt upgrade & apt install curl &pip install --no-cache-dir --upgrade -r /tmp/requirements.txt

#COPY ./app /code/app
#CMD [ "bash" ]
#CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]