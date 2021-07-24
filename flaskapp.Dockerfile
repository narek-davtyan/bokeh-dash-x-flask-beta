FROM continuumio/miniconda
ENV BK_VERSION=2.2.2
ENV PY_VERSION=3.8
ENV NUM_PROCS=4
ENV BOKEH_RESOURCES=cdn
# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

RUN apt-get install git bash
RUN git clone https://github.com/narek-davtyan/bokeh-dash-x-flask-beta.git
RUN cd bokeh-dash-x-flask-beta
RUN conda install --yes --quiet python=${PY_VERSION} numpy packaging pandas bokeh=${BK_VERSION} pillow pyparsing python-dateutil pytz pyyaml six tornado typing-extensions "xlrd==1.2.0"
RUN conda clean -ay
RUN pip install Flask gunicorn wordcloud==1.8.0


# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# EXPOSE 8080
EXPOSE 5006
ENV PORT 5006

# CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app


CMD python flaskapp/app.py
# CMD bokeh serve bokeh-vis-bd-x/ --port 8080 \
#     --allow-websocket-origin="*" \
#     --num-procs=${NUM_PROCS}
    