FROM python:3.8

WORKDIR /light_side_fastapi
COPY . light_side_fastapi/

RUN pip install light_side --no-cache-dir --upgrade
RUN pip install -r light_side_fastapi/requirements.txt --no-cache-dir

CMD ["python", "light_side_fastapi/service.py"]
