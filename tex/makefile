#
#
#

all: output clean

#
#
#

output:
	pdflatex paper
	bibtex paper
	pdflatex paper
	pdflatex paper

#
#
#

clean: output
	rm *.bbl
	rm *.brf *.aux *.log *.blg
