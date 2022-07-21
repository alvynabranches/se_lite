FROM python:3.10

COPY requirements.txt requirements.txt

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y ffmpeg
RUN /usr/local/bin/python3 -m pip install --upgrade pip setuptools
RUN /usr/local/bin/python3 -m pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD [ "streamlit", "run", "main.py" ]