import random
import json
import os
import math

# --- Helper Function: Sample JSONL File by Ratio (Same as the Original Code Provided) ---

def sample_and_save_jsonl(input_file_path: str, output_file_path: str, sample_ratio: float, seed: int = 42):
    """
    Randomly sample a JSONL file by a specified ratio and save the sampled results to a new JSONL file.

    Args:
        input_file_path (str): Path to the input JSONL file.
        output_file_path (str): Path to the output JSONL file where sampled results will be saved.
        sample_ratio (float): Sampling ratio (0.0 to 1.0).
        seed (int): Seed for fixing the random number generator of the random module.
    """
    # 1. Check if the input file exists
    if not os.path.exists(input_file_path):
        print(f"❌ Error: Input file not found, path: {input_file_path}")
        return

    all_lines = []
    # 2. Read all lines from the file
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            # Filter out empty lines or lines with only whitespace characters
            all_lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"❌ Error occurred while reading the file: {e}")
        return

    total_lines = len(all_lines)
    if total_lines == 0:
        print(f"⚠️ Warning: No valid data lines in file {input_file_path}, no sampling needed.")
        # Ensure the target sampled file is empty if the original file is empty
        if os.path.exists(output_file_path):
            os.remove(output_file_path)
        return

    # 3. Calculate the number of samples to take
    # Use math.ceil to ensure at least 1 record is sampled
    num_samples = math.ceil(total_lines * sample_ratio)
    
    # Ensure the number of samples does not exceed the total number of lines
    if num_samples > total_lines:
        num_samples = total_lines
        
    print(f"--- Starting sampling: {input_file_path} ---")
    print(f"Total lines: {total_lines}")
    print(f"Sampling ratio: {sample_ratio*100}%")
    print(f"Calculated number of samples: {num_samples}")

    # 4. Fix the random seed
    random.seed(seed)

    # 5. Perform sampling using random.sample
    sampled_lines = random.sample(all_lines, num_samples)

    # 6. Save the sampled results to a new JSONL file
    try:
        # Create the target directory if it does not exist
        output_dir = os.path.dirname(output_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for line in sampled_lines:
                f.write(line + '\n')
                
        print(f"✅ Sampling successful! A total of {len(sampled_lines)} records saved to file: {output_file_path}")

    except Exception as e:
        print(f"❌ Error occurred while saving the file: {e}")
        
    # 7. Print the specific sampled records (for display)
    print("\n--- Specific sampled records (first 5 or all) ---")
    for i, record in enumerate(sampled_lines[:5]):
        # Try to load JSON for pretty printing
        try:
            parsed_json = json.loads(record)
            # Assume records may contain 'id' or 'image' fields
            identifier = parsed_json.get('id') or parsed_json.get('image') or 'N/A'
            print(f"Record {i+1} ID/Image: {identifier}")
        except json.JSONDecodeError:
            print(f"Record {i+1} (unparseable): {record[:50]}...")
    print("---------------------------------------------\n")


# --- New Function: Merge Multiple JSONL Files ---

def merge_jsonl_files(input_files: list[str], output_file_path: str):
    """
    Merge multiple JSONL files into a new JSONL file in sequence.

    Args:
        input_files (list[str]): List of paths to input JSONL files to be merged (in order).
        output_file_path (str): Path to the output JSONL file where merged results will be saved.
    """
    total_merged_lines = 0
    
    # Create the target directory if it does not exist
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"\n--- Starting merging files into: {output_file_path} ---")

    try:
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            for i, input_file in enumerate(input_files):
                if not os.path.exists(input_file):
                    print(f"⚠️ Warning: Input file not found, path: {input_file}, skipping this file.")
                    continue
                
                line_count = 0
                try:
                    with open(input_file, 'r', encoding='utf-8') as infile:
                        for line in infile:
                            stripped_line = line.strip()
                            if stripped_line:  # Filter empty lines, only write valid records
                                outfile.write(stripped_line + '\n')
                                line_count += 1
                                total_merged_lines += 1
                except Exception as e:
                    print(f"❌ Error occurred while reading file {input_file}: {e}")
                    continue
                    
                print(f"  [{i+1}/{len(input_files)}] Merged file '{input_file}', containing {line_count} records.")
                
        print(f"\n✅ Merging successful! A total of {total_merged_lines} records saved to file: {output_file_path}")

    except Exception as e:
        print(f"❌ Error occurred while saving the merged file: {e}")


# ======================================================================
# --- Configuration Parameters ---

# 1. COCO Sampling Configuration
COCO_INPUT_FILE = "datasets/coco_cambrian.jsonl"
COCO_SAMPLED_FILE = "datasets/coco_cambrian_4per.jsonl"
SAMPLE_RATIO = 0.04
RANDOM_SEED = 42

# 2. Merged File Path Configuration
TEXTVQA_FILE = "datasets/textvqa_cambrian.jsonl"
OCR_VQA_FILE = "datasets/ocr_vqa_cambrian.jsonl"
MERGED_OUTPUT_FILE = "datasets/textvqa_ocrvqa_cambrian.jsonl"

# 3. Define the final merging order: chartqa -> coco_4per -> ocr_vqa
FILES_TO_MERGE = [
    TEXTVQA_FILE,
    # COCO_SAMPLED_FILE,  # This is the file generated by sampling
    OCR_VQA_FILE
]

def main():
    """Main execution function, sequentially performs sampling and merging operations."""
    
    print("--- Task started ---")

    # Step 1: Sample 4% of coco_cambrian.jsonl
    # sample_and_save_jsonl(COCO_INPUT_FILE, COCO_SAMPLED_FILE, SAMPLE_RATIO, RANDOM_SEED)

    # Step 2: Merge files in the specified order
    merge_jsonl_files(FILES_TO_MERGE, MERGED_OUTPUT_FILE)
    
    print("\n--- Task completed ---")

# --- Run the main function ---
if __name__ == "__main__":
    main()