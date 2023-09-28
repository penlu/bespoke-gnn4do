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
    for i in $(cat $STATPATH | grep volta | grep RUNNI | cut -d " " -f 2 | sort -n) ; do
        echo $i $(tail -n 1 *$i.out)
        echo $i $(head -n 2 *$i.out | tail -n 1 | grep flags)
    done
done
