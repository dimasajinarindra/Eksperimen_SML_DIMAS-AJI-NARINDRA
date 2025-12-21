FROM python:3.12-slim
WORKDIR /app
COPY MLProject/ /app/
RUN pip install mlflow==2.19.0 scikit-learn pandas dagshub
CMD ["python", "modelling.py"]
