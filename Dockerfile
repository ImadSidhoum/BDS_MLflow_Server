FROM python:3.8-slim-buster
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt 
RUN pip install opencv-python-headless
COPY . /app
EXPOSE 5001
ENTRYPOINT ["uvicorn", "Serve:app","--host", "0.0.0.0" , "--port", "5001", "--reload"]
#docker run -it  -p 5001:5001 --mount type=bind,source=$(pwd),target=/app land95/mlflow-server:0.1
