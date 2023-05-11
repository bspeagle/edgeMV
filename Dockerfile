FROM python:3.7.7-alpine

WORKDIR /usr/src/app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt
RUN pip3 install --upgrade -r requirements.txt

COPY . ./

CMD ["python3","-u","src/service.py"]