#!/bin/bash
mkdir -p blastdb
cd blastdb
curl -O ftp://ftp.ncbi.nlm.nih.gov/blast/db/swissprot.tar.gz
tar -xzvf swissprot.tar.gz
rm swissprot.tar.gz