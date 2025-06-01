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

echo "🧹 Cleaning up..."
rm -rf ncbi-blast-2.14.1+*
rm ncbi-blast-2.14.1+-x64-linux.tar.gz

echo "✅ psiblast is ready."

# Optional: also setup the database
echo "🔄 Setting up SwissProt BLAST DB..."
mkdir -p blastdb
cd blastdb
docker run --rm -v $(pwd):/blastdb ncbi/blast update_blastdb.pl --decompress --passive swissprot