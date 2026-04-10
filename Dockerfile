FROM public.ecr.aws/docker/library/python:3.10

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY . .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r server/requirements.txt

EXPOSE 7860
CMD ["python", "-m", "server.app"]