import json
import os
from typing import List, Dict, Any, IO

def filter_multimodal_datasets(input_file: str, output_dir: str, target_datasets: List[str], output_filename_base: str = "cambrian"):
    """
    Filters records in a JSONL file and splits them into separate JSONL files 
    based on the dataset name found as a substring in the 'image' field.
    
    Args:
        input_file: Path to the input .jsonl file.
        output_dir: Directory where the filtered output files will be saved.
        target_datasets: A list of dataset names (substrings) to keep, e.g., ['ocr_vqa', 'chartqa', 'coco'].
        output_filename_base: Base name for the output files (e.g., 'ocr_vqa_cambrian.jsonl').
    """
    
    # Convert target dataset names to lowercase for case-insensitive matching
    target_datasets_lower = [name.lower() for name in target_datasets]
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to hold the file handles for writing each dataset
    output_files: Dict[str, IO] = {}
    # Dictionary to count records written for each dataset
    record_counts: Dict[str, int] = {name: 0 for name in target_datasets}
    
    # Prepare output file paths and open handles
    try:
        for dataset_name in target_datasets:
            filename = f"{dataset_name.lower()}_{output_filename_base}.jsonl"
            filepath = os.path.join(output_dir, filename)
            # Open file handles for writing in append mode (a), ensuring unicode support
            output_files[dataset_name.lower()] = open(filepath, 'w', encoding='utf-8')
            print(f"Prepared output file: {filepath}")

    except Exception as e:
        print(f"Error opening output files: {e}")
        return

    print(f"Starting to process file: {input_file}")

    # Read the JSONL file line by line
    records_processed = 0
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                records_processed += 1
                if not line.strip():
                    continue

                try:
                    item = json.loads(line)
                    
                    if not isinstance(item, dict) or 'image' not in item:
                        continue
                    
                    image_path: str = item['image'] if item.get('image') is not None else ""
                    image_path_lower = image_path.lower()
                    
                    # Core filtering and routing logic
                    for dataset_name in target_datasets_lower:
                        if dataset_name in image_path_lower:
                            # Found a match, route the record to the specific dataset file
                            
                            # Construct a new item with the required fields
                            new_item = {
                                "id": item.get('id'),
                                "image": item['image'],
                                "conversations": item.get("conversations")
                            }
                            
                            # Write the record to the corresponding file handle
                            json_line = json.dumps(new_item, ensure_ascii=False) + '\n'
                            output_files[dataset_name].write(json_line)
                            
                            record_counts[dataset_name] += 1
                            break
                    
                    if records_processed % 100000 == 0:
                        print(f"Processed {records_processed} records.")


                except json.JSONDecodeError as e:
                    print(f"Warning: JSON decoding error on line {line_num}: {e}. Skipping line.")
                except Exception as e:
                    print(f"Warning: An unknown error occurred while processing line {line_num}: {e}. Skipping line.")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
    except Exception as e:
        print(f"Error during file reading: {e}")
    finally:
        # Close all open output file handles
        for handle in output_files.values():
            handle.close()

    print("-" * 40)
    print(f"Processing complete. Total records processed: {records_processed}")
    print("Records saved per dataset:")
    for name, count in record_counts.items():
        print(f"  - {name.upper()}: {count} records saved to {name}_{output_filename_base}.jsonl")
    print("-" * 40)

# --- Configuration Parameters ---
INPUT_FILE = 'datasets/Cambrian737k.jsonl'
OUTPUT_DIR = 'datasets' 

TARGET_DATASETS_TO_KEEP = ['ocr_vqa', 'chartqa', 'textvqa', 'coco']

if __name__ == "__main__":
    filter_multimodal_datasets(INPUT_FILE, OUTPUT_DIR, TARGET_DATASETS_TO_KEEP,"cambrian")
