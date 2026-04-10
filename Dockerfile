FROM public.ecr.aws/docker/library/python:3.10
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r server/requirements.txt
EXPOSE 7860
# Command to start the Support Engine
CMD ["python", "server/app.py"]