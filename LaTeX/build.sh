#/bin/bash
set -e
if [ ! -e tmp ]; then
  mkdir tmp
fi
if [[ [$(uname) == "Linux"] || [$(uname) == "Darwin"] ]]; then
  LATEX="exlatex"
else
  LATEX="pdflatex"
fi
echo "Building with "$LATEX
"$LATEX" -output-directory tmp line-tpami
bibtex tmp/line-tpami
"$LATEX" -output-directory tmp -interaction=nonstopmode line-tpami
"$LATEX" -output-directory tmp -interaction=nonstopmode line-tpami

