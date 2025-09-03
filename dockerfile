FROM python:3.13
COPY . /project
WORKDIR /project
RUN pip install uv
RUN uv pip install --system -e .[compile]
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
