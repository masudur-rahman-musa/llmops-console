# ADAPT to the stack. Multi-stage: build in a fat image, ship a slim one.
# Copy src/ and the dependency manifest only; .dockerignore keeps the rest out.
# Run as a non-root user. Expose the port. Define a HEALTHCHECK against /healthz.
FROM ADAPT:base AS build
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN echo "ADAPT: install dependencies" && exit 1
COPY src/ ./src/
FROM ADAPT:slim
WORKDIR /app
COPY --from=build /app /app
USER 1000
EXPOSE 8000
HEALTHCHECK CMD echo "ADAPT: curl -f http://localhost:8000/healthz" || exit 1
CMD ["echo", "ADAPT: the run command"]
