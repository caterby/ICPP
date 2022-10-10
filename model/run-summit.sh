#!/bin/zsh
set -eu

# RUN SUMMIT SH
# Creates an LSF job file and submits it

# Absolute path to the directory containing this script
export THIS=${${0:h}:A}

cd $THIS
TMPFILE=$( mktemp output.txt.XXX )
export OUTPUT=$PWD/$TMPFILE

m4 -P job-summit.sh.m4 > job-summit.sh

bsub job-summit.sh | read MESSAGE
echo $MESSAGE
# Pull out 2nd word without characters '<' and '>'
JOB_ID=${${(z)MESSAGE}[3]}

[[ ${JOB_ID} != "" ]] || abort "bsub failed!"

declare JOB_ID

echo Waiting ...
bwait -w "done($JOB_ID)"
