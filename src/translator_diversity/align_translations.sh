#!/bin/bash
#PBS -N align_translations
#PBS -l select=1:ncpus=1:ngpus=1:gpu_mem=50gb:mem=50gb
#PBS -o job_logs/outputs
#PBS -e job_logs/errors

# Define all book-version pairs
declare -A book_versions=(
    #["1984"]="bussetti dilon martin temprano vazquez"
    #["alice_wonderland"]="barba gallo maristany rodriguez torres"
    #["dracula"]="casas molina montalban rodriguez"
    #["great_gatsby"]="alvarez carral cohen lopez navarro"
    #["huckleberry_finn"]="larrinaga rolfe unknown"
    #["moby_dick"]="pezzoni valverde velasco"
    #["pride_prejudice"]="ibanez rodriguez salis"
    #["ulysses"]="salas valverde venegas"
    #["wuthering_heights"]="castillo damondville deluaces martin santervas"
    ["2br02b"]="guerberof gandolfo maldonado unknown"
)


export TMPDIR=$SCRATCHDIR
VENVDIR=/storage/brno2/home/maldonj/creativity_detection/venv
DATADIR=/storage/brno2/home/maldonj/creativity_detection/data
SRCDIR=/storage/brno2/home/maldonj/creativity_detection/src
EXTTOOLSDIR=/storage/brno2/home/maldonj/creativity_detection/external_tools

source $VENVDIR/bin/activate
export CC=gcc
export CXX=G++
export LASER="/storage/brno2/home/maldonj/creativity_detection/external_tools/LASER"

#rm -rf $DATADIR/aligned_books/alignment_files/vecalign_files
#rm -rf $DATADIR/aligned_books/alignment_files/alignments
mkdir $DATADIR/aligned_books
mkdir $DATADIR/aligned_books/alignment_files
mkdir $DATADIR/aligned_books/alignment_files/vecalign_files
mkdir $DATADIR/aligned_books/alignment_files/alignments

# Process each book
for book in "${!book_versions[@]}"; 
do
    echo "Processing book: $book"
    
    rm -rf $DATADIR/aligned_books/alignment_files/vecalign_files/$book
    rm -rf $DATADIR/aligned_books/alignment_files/alignments/$book
    mkdir $DATADIR/aligned_books/alignment_files/vecalign_files/$book
    mkdir $DATADIR/aligned_books/alignment_files/alignments/$book
    
    # Convert versions string to array
    IFS=' ' read -ra versions <<< "${book_versions[$book]}"
    
    # Get overlaps and embeddings for original
    $VENVDIR/bin/python3 $EXTTOOLSDIR/vecalign/overlap.py -i $DATADIR/books/$book/original.txt -o $DATADIR/aligned_books/alignment_files/vecalign_files/$book/original_overlaps
    $VENVDIR/bin/python3 $EXTTOOLSDIR/LASER/source/embed.py --input $DATADIR/aligned_books/alignment_files/vecalign_files/$book/original_overlaps --encoder $EXTTOOLSDIR/laser_models/laser2.pt --spm-model $EXTTOOLSDIR/laser_models/laser2.spm --output $DATADIR/aligned_books/alignment_files/vecalign_files/$book/original_overlaps.emb
    
    # Align each translation to the original individually
    for version in "${versions[@]}";
    do
        echo "  Processing version: $version"
        $VENVDIR/bin/python3 $EXTTOOLSDIR/vecalign/overlap.py -i $DATADIR/books/$book/$version.txt -o $DATADIR/aligned_books/alignment_files/vecalign_files/$book/$version\_overlaps
        $VENVDIR/bin/python3 $EXTTOOLSDIR/LASER/source/embed.py --input $DATADIR/aligned_books/alignment_files/vecalign_files/$book/$version\_overlaps --encoder $EXTTOOLSDIR/laser_models/laser2.pt --spm-model $EXTTOOLSDIR/laser_models/laser2.spm --output $DATADIR/aligned_books/alignment_files/vecalign_files/$book/$version\_overlaps.emb
        $VENVDIR/bin/python3 $EXTTOOLSDIR/vecalign/vecalign.py --alignment_max_size 8 --src $DATADIR/books/$book/original.txt --tgt $DATADIR/books/$book/$version.txt --src_embed $DATADIR/aligned_books/alignment_files/vecalign_files/$book/original_overlaps $DATADIR/aligned_books/alignment_files/vecalign_files/$book/original_overlaps.emb --tgt_embed $DATADIR/aligned_books/alignment_files/vecalign_files/$book/$version\_overlaps $DATADIR/aligned_books/alignment_files/vecalign_files/$book/$version\_overlaps.emb >> $DATADIR/aligned_books/alignment_files/alignments/$book/$version\_raw.txt
        
        # Parse alignment for visual evaluation
        $VENVDIR/bin/python3 $SRCDIR/translator_diversity/parse_vecalign_alignment.py --alignment $DATADIR/aligned_books/alignment_files/alignments/$book/$version\_raw.txt --source $DATADIR/books/$book/original.txt --target $DATADIR/books/$book/$version.txt --output $DATADIR/aligned_books/alignment_files/alignments/$book/$version.csv
    done
    
    # Merge all of the alignments for this book
    $VENVDIR/bin/python3 $SRCDIR/translator_diversity/merge_alignments.py --alignments $DATADIR/aligned_books/alignment_files/alignments/$book --book $DATADIR/books/$book --output $DATADIR/aligned_books/$book.csv
    
    echo "Completed book: $book"
done

echo "All books processed!"