# Rebuild every table, figure and number from bench/results, then the PDF.
paper.pdf: paper.tex references.bib make_assets.py $(wildcard ../bench/results/*/*/*/*/*.json)
	cd .. && .venv/bin/python paper/make_assets.py > /dev/null
	latexmk -pdf -interaction=nonstopmode -quiet paper.tex

preprint.pdf: preprint.tex paper.pdf
	latexmk -pdf -interaction=nonstopmode -quiet preprint.tex

# arXiv source bundle: the preprint with its bibliography already compiled
arxiv: preprint.pdf
	tar -czf arxiv.tar.gz preprint.tex paper.tex preprint.bbl acl.sty acl_natbib.bst references.bib generated figures/runs.pdf

clean:
	latexmk -C paper.tex
	latexmk -C preprint.tex

supplementary:
	cd .. && git archive --format=zip -o paper/supplementary.zip HEAD -- . ":(exclude)paper"

.PHONY: clean supplementary arxiv
