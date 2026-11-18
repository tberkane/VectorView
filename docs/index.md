# VectorView

**Text search over street-level imagery.**

VectorView finds images of a visual feature, described in plain words, among the Street View images
of an area: "a discarded tire on a lawn", "standing water in a roadside ditch", "an overflowing
dumpster". It was built to help map possible mosquito breeding sites, but works for any feature
visible from the street.

1. **Collect** the images of an area, along the roads with the Street View Static API (with a hard
   per-key image cap) or for every panorama in the area with the `streetlevel` library.
2. **Embed** each image, its grid tiles and its patch regions with SigLIP-2.
3. **Expand** the query with an LLM into paraphrases, scene context and look-alikes.
4. **Rank** the images, then **rerank** the top ones with a vision-language model that answers
   yes/no with a reason and a bounding box.
5. **Verify** results in a local web app and **refine** the query from your verdicts.

Start with the [quickstart](quickstart.md); [how it works](concepts.md) explains the ranking and
its limitations.

## Install

```bash
# Install the PyTorch build for your machine first: https://pytorch.org/get-started/locally/
pip install "vectorview[encoding,llm,web] @ git+https://github.com/tberkane/VectorView@v0.1.0"
```

Python 3.11 or later, on Linux or macOS. Add the `scraper` extra for the `streetlevel` source. The
`vectorview` package on PyPI is unrelated; install from GitHub as above.

## License and citation

Apache License 2.0. To cite VectorView, use the repository's
[CITATION.cff](https://github.com/tberkane/VectorView/blob/main/CITATION.cff).
