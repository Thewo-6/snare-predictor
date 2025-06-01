#!/bin/bash
set -e

echo "🔄 Downloading and extracting psiblast..."

# Download Linux-compatible BLAST+ tools
curl -LO https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/ncbi-blast-*-x64-linux.tar.gz

# Extract and move only psiblast
tar -xzf ncbi-blast-*-x64-linux.tar.gz
mv ncbi-blast-*/bin/psiblast bin/psiblast
chmod +x bin/psiblast

# Clean up
rm -rf ncbi-blast-*
rm ncbi-blast-2.14.1+-x64-linux.tar.gz

echo "✅ psiblast ready!"

echo "🔄 Setting up SwissProt DB..."
mkdir -p blastdb
cd blastdb
docker run --rm -v $(pwd):/blastdb ncbi/blast update_blastdb.pl --decompress --passive swissprot
echo "✅ SwissProt DB ready!"