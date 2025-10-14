import re

# Define placeholders for dataset paths

OCR_VQA ={
    "annotation_path": "../datasets/ocr_vqa_cambrian.jsonl",
    "data_path": "../datasets",
}

COCO ={
    "annotation_path": "../datasets/coco_cambrian.jsonl",
    "data_path": "../datasets",
}

CHARTQA ={
    "annotation_path": "../datasets/chartqa_cambrian.jsonl",
    "data_path": "../datasets",
}

data_dict = {
    "ocr_vqa": OCR_VQA,
    "coco": COCO,
    'chartqa': CHARTQA,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["ocr_vqa"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
