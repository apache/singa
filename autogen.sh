#!/bin/sh

mkdir -p ./config;
aclocal;
autoreconf -f -i;
automake;
