all: clean tesis

tesis:
	pdflatex tesis.tex
	bibtex tesis.aux
	pdflatex tesis.tex
	pdflatex tesis.tex

clean:
	rm -f *.aux *.log *.toc tesis.pdf


