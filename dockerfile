FROM python:3.13
WORKDIR /project
COPY . .
RUN pip install uv
RUN uv pip install --system -e .[compile]
EXPOSE 8006
CMD ["uvicorn", "app.main:create_app", "--host", "0.0.0.0", "--port", "8006", "--factory"]