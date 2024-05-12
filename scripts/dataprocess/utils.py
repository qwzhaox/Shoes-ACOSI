OPINION_IDX = 3
IMPL_EXPL_IDX = 4

def get_dataset(dataset_folder_path):
    dataset = []

    for split in ["train", "test", "dev"]:
        with open(f"{dataset_folder_path}/{split}.txt") as f:
            dataset.extend(f.readlines())

    return dataset