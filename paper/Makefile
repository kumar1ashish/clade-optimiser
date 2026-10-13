# Rebuild every table, figure and number from bench/results, then the PDF.
paper.pdf: paper.tex references.bib make_assets.py $(wildcard ../bench/results/*/*/*/*/*.json)
	cd .. && .venv/bin/python paper/make_assets.py > /dev/null
	latexmk -pdf -interaction=nonstopmode -quiet paper.tex

preprint.pdf: preprint.tex paper.pdf
	latexmk -pdf -interaction=nonstopmode -quiet preprint.tex

# arXiv source bundle: one top-level file (the preprint), its compiled bibliography, styles, tables and figure
arxiv: preprint.pdf
	rm -rf arxiv && mkdir -p arxiv/generated arxiv/figures
	{ echo '\def\preprintmode{}'; cat paper.tex; } > arxiv/main.tex
	cp preprint.bbl arxiv/main.bbl
	cp acl.sty acl_natbib.bst arxiv/ && cp generated/*.tex arxiv/generated/ && cp figures/runs.pdf arxiv/figures/
	tar -czf arxiv.tar.gz -C arxiv . && rm -rf arxiv

clean:
	latexmk -C paper.tex
	latexmk -C preprint.tex

supplementary:
	cd .. && git archive --format=zip -o paper/supplementary.zip HEAD -- . ":(exclude)paper"

.PHONY: clean supplementary arxiv
