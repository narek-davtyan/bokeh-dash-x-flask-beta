FROM continuumio/miniconda
ENV BK_VERSION=2.2.2
ENV PY_VERSION=3.8
ENV NUM_PROCS=4
ENV BOKEH_RESOURCES=cdn
RUN apt-get install git bash
RUN git clone https://github.com/narek-davtyan/bokeh-vis-bd-x.git
RUN cd bokeh-vis-bd-x
RUN conda install --yes --quiet python=${PY_VERSION} numpy packaging pandas bokeh=${BK_VERSION} pillow pyparsing python-dateutil pytz pyyaml six tornado typing-extensions "xlrd==1.2.0"
RUN conda clean -ay
RUN pip install wordcloud==1.8.0 pandarallel==1.5.1
EXPOSE 8080
CMD bokeh serve bokeh-vis-bd-x/ --port 8080 \
    --allow-websocket-origin="*" \
    --num-procs=${NUM_PROCS}
    