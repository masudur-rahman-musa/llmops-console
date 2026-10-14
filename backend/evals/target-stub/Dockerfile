# The kind e2e's throwaway eval target. `main.py` is stdlib-only (a project rule: no new
# dependency), so this image needs nothing installed beyond the interpreter itself -
# built and `kind load docker-image`d fresh on every workflow run, alongside the
# gateway's own image, per an earlier change decision D.
FROM python:3.12-slim

WORKDIR /app
COPY main.py ./

USER 1000
EXPOSE 8090

CMD ["python", "main.py"]
