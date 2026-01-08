#!/bin/bash
#PBS -N epub_conversion
#PBS -l select=1:ncpus=1 
#PBS -o job_logs/outputs
#PBS -e job_logs/errors

declare -A book_versions=(
    ["1984"]="bussetti dilon martin temprano vazquez"
    ["alice_wonderland"]="barba gallo maristany rodriguez torres"
    ["dracula"]="casas molina montalban rodriguez"
    ["great_gatsby"]="alvarez carral cohen lopez navarro"
    ["huckleberry_finn"]="larrinaga rolfe unknown"
    ["moby_dick"]="pezzoni valverde velasco"
    ["pride_prejudice"]="ibanez rodriguez salis"
    ["ulysses"]="salas valverde venegas"
    ["wuthering_heights"]="castillo damondville deluaces martin santervas"
)

export TMPDIR=$SCRATCHDIR

VENVDIR=/storage/brno2/home/maldonj/creativity_detection/venv
DATADIR=/storage/brno2/home/maldonj/creativity_detection

for book in "${!book_versions[@]}";
do
    # Convert versions string to array
    IFS=' ' read -ra versions <<< "${book_versions[$book]}"

    for version in "${!versions[@]}";
    do
        $VENVDIR/bin/python3 $DATADIR/src/translator_diversity/convert_epub.py --input $DATADIR/data/books/$book/01_original_files/$version.epub --output $DATADIR/books/$book/02_raw_text/$version.txt
    done
done