#!/bin/bash
set -e

echo "🔄 Creating bin folder..."
mkdir -p bin

echo "🔄 Downloading Linux-compatible BLAST+ tools..."
curl -LO https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.14.1/ncbi-blast-2.14.1+-x64-linux.tar.gz

echo "📦 Extracting..."
tar -xzf ncbi-blast-2.14.1+-x64-linux.tar.gz
mv ncbi-blast-2.14.1+/bin/psiblast bin/
chmod +x bin/psiblast

echo "🧹 Cleaning up BLAST+ package..."
rm -rf ncbi-blast-2.14.1+*

echo "✅ psiblast is ready."

# 🔄 Now download SwissProt directly
echo "🔽 Downloading SwissProt BLAST DB..."
mkdir -p blastdb
cd blastdb
curl -O ftp://ftp.ncbi.nlm.nih.gov/blast/db/swissprot.tar.gz
tar -xzf swissprot.tar.gz
rm swissprot.tar.gz

echo "✅ SwissProt BLAST DB ready!"