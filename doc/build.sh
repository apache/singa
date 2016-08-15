#!/bin/bash

SPHINXBUILD="sphinx-build"
BUILDDIR="_build"
LANG_ARR=(en zh jp kr)

if [ "$1"x = "clean"x ]; then
	rm -rf $BUILDDIR/*
	echo "clean up $BUILDDIR"
fi

if [ "$1"x = "html"x ]; then
	for (( i=0; i<${#LANG_ARR[@]}; i++)) do
		echo "building language ${LANG_ARR[i]} ..."
		$SPHINXBUILD -b html -c . -d $BUILDDIR/doctree ${LANG_ARR[i]} $BUILDDIR/html/${LANG_ARR[i]}
	done
	echo "<script language=\"javascript\" type=\"text/javascript\">window.location.href='en/index.html';</script>" > $BUILDDIR/html/index.html
fi
