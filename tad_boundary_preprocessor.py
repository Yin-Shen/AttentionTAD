#!/usr/bin/env python3

import os
import random
import argparse
from Bio import SeqIO
from pybedtools import BedTool
import numpy as np
from sklearn.model_selection import train_test_split

def read_fasta(fasta_file):
    return {rec.id: str(rec.seq).upper() for rec in SeqIO.parse(fasta_file, "fasta")}

def get_sequence(chrom, start, end, genome_dict):
    return genome_dict[chrom][start:end]

def get_valid_positive_samples(bed_file, genome_dict):
    positive_samples = []
    
    print("Loading BED file...")
    bed = BedTool(bed_file)
    
    for interval in bed:
        chrom = interval.chrom
        start = interval.start
        end = interval.end
        
        if chrom in genome_dict:
            if end <= len(genome_dict[chrom]):
                positive_samples.append((chrom, start, end))
            else:
                print(f"Warning: Skipping sample that exceeds chromosome length - {chrom}:{start}-{end}")
    
    return positive_samples

def save_negative_samples_bed(negative_samples, output_bed_file):
    print(f"Saving negative samples to {output_bed_file}...")
    with open(output_bed_file, 'w') as f:
        for chrom, start, end in negative_samples:
            f.write(f"{chrom}\t{start}\t{end}\n")
    print(f"Saved {len(negative_samples)} negative samples to BED file")

def generate_balanced_negative_samples(positive_samples, genome_dict):
    negative_samples = []
    chroms = list(genome_dict.keys())
    
    num_negative = len(positive_samples)
    print(f"Generating {num_negative} negative samples to match positive samples...")
    
    attempts = 0
    max_attempts = num_negative * 10
    
    while len(negative_samples) < num_negative and attempts < max_attempts:
        chrom = random.choice(chroms)
        chrom_length = len(genome_dict[chrom])
        
        if chrom_length < 10000:
            continue
        
        start = random.randint(0, chrom_length - 10000)
        end = start + 10000
        
        if not any(p_start <= start < p_end or p_start < end <= p_end 
                  or (start <= p_start and end >= p_end)
                  for p_chrom, p_start, p_end in positive_samples if p_chrom == chrom):
            sample = (chrom, start, end)
            if sample not in negative_samples:
                negative_samples.append(sample)
        
        attempts += 1
    
    if len(negative_samples) < num_negative:
        print(f"Warning: Could only generate {len(negative_samples)} unique negative samples")
        positive_samples = random.sample(positive_samples, len(negative_samples))
        print(f"Randomly sampled positive samples to match negative sample count")
    
    return positive_samples, negative_samples

def one_hot_encode(sequence):
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], 'N': [0,0,0,0]}
    try:
        return np.array([mapping[base.upper()] for base in sequence])
    except KeyError as e:
        print(f"Error in one_hot_encode: Invalid base '{e.args[0]}' found in sequence.")
        print(f"Sequence: {sequence}")
        raise

def process_data(bed_file, fasta_file, output_file, negative_bed_file):
    genome_dict = read_fasta(fasta_file)
    
    positive_samples = get_valid_positive_samples(bed_file, genome_dict)
    print(f"Found {len(positive_samples)} valid positive samples")
    
    positive_samples, negative_samples = generate_balanced_negative_samples(
        positive_samples, genome_dict)
    
    save_negative_samples_bed(negative_samples, negative_bed_file)
    
    print(f"Final balanced dataset:")
    print(f"Positive samples: {len(positive_samples)}")
    print(f"Negative samples: {len(negative_samples)}")
    
    X, y = [], []
    
    all_samples = [(sample, 1) for sample in positive_samples] + \
                 [(sample, 0) for sample in negative_samples]
    
    print("Processing sequences...")
    for i, (sample, label) in enumerate(all_samples):
        chrom, start, end = sample
        if chrom not in genome_dict:
            print(f"Warning: Chromosome {chrom} not found in the genome dictionary. Skipping this sample.")
            continue
            
        try:
            seq = get_sequence(chrom, start, end, genome_dict)
            X.append(one_hot_encode(seq))
            y.append(label)
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

        if i % 1000 == 0:
            print(f"Processed {i} samples...")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Final processed samples: {len(X)}")
    print(f"Positive samples: {np.sum(y)}")
    print(f"Negative samples: {len(y) - np.sum(y)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    np.savez(output_file, 
             X_train=X_train, X_test=X_test, 
             y_train=y_train, y_test=y_test)
    
    print(f"Data preprocessed and saved to {output_file}")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

def main():
    parser = argparse.ArgumentParser(description="Process TAD boundary data for a specific cell line.")
    parser.add_argument("cell_line", help="Name of the cell line to process (e.g., GM12878)")
    args = parser.parse_args()

    cell_line = args.cell_line
    bed_file = f"{cell_line}_TAD_boundaries.bed"
    fasta_file = "hg19.fa"
    output_file = f"{cell_line}_tad_boundary_data.npz"
    negative_bed_file = f"{cell_line}_negative_samples.bed"

    if not os.path.exists(bed_file):
        print(f"Error: BED file '{bed_file}' not found.")
        exit(1)

    if not os.path.exists(fasta_file):
        print(f"Error: FASTA file '{fasta_file}' not found.")
        exit(1)

    process_data(bed_file, fasta_file, output_file, negative_bed_file)

if __name__ == "__main__":
    main()
