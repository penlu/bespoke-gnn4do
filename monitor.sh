#!/usr/bin/env bash

set -e

STATPATH=~/recent_LLstats.txt

while true ; do
    sleep 5
    LLstat > $STATPATH
    date
    cat $STATPATH | grep myau | wc -l
    cat $STATPATH | grep RUNNI
    ls -lart | tail -n 20
    for i in $(cat $STATPATH | cut -d " " -f 2) ; do
        tail -n 1 *$i.out
    done
done
