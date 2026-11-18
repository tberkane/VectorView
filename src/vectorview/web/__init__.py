"""Local web app: draw a region, collect and embed its images, then search them with free text.

A single-user FastAPI app (``vectorview serve``). State lives in SQLite; embeddings in numpy
caches; work runs in one background worker (`vectorview.web.worker`).
"""
