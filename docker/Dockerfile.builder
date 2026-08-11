FROM iaxl-dev AS builder
ENV DEBIAN_FRONTEND=noninteractive

FROM scratch AS exporter
COPY --from=builder /install /
