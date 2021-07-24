FROM continuumio/miniconda3

RUN conda install -y nomkl bokeh numpy pandas

VOLUME '/app'

EXPOSE 5006

ENTRYPOINT ["bokeh","serve","/app/bokehapp","--allow-websocket-origin=*"]