# Local Spark runtime that mirrors Dataproc 2.2 (Spark 3.5, Java 17, Python 3).
FROM apache/spark:3.5.3-scala2.12-java17-python3-ubuntu
USER root
RUN pip install --no-cache-dir numpy==1.26.4
WORKDIR /app
