from glob import glob
from natsort import natsorted
import os
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image


def get_split_paths(dataset_path, split):
    is_toy = split == 'toy'
    if split == 'toy': # a toy split to try things out quickly
        split = 'train'
        limit_at = 1
    else:
        limit_at = -1
    
    video_folders = natsorted(glob(f'{dataset_path}/{split}/*/'))
    video_folders = video_folders[:limit_at] if limit_at > 0 else video_folders
    rgb_frame_paths = []
    event_frame_paths = []
    frame_index_in_video = []
    print(len(video_folders), "videos found in the", split, "set.")
    for folder in tqdm(video_folders):
        rgb_paths = natsorted(glob(f'{folder}/RGB/*.jpg'))
        event_paths = natsorted(glob(f'{folder}/Event/Frames/*.png'))
        rgb_frame_paths.extend(rgb_paths)
        event_frame_paths.extend(event_paths)
        frame_index_in_video.extend(list(range(len(rgb_paths))))

    if is_toy:
        rgb_frame_paths = rgb_frame_paths[:440]
        event_frame_paths = event_frame_paths[:440]
        frame_index_in_video = frame_index_in_video[:440]

    assert(len(rgb_frame_paths) == len(event_frame_paths)), "Mismatch in number of RGB and Event frames"
    return list(zip(rgb_frame_paths, event_frame_paths, frame_index_in_video))


def read_coordinates(file_path):
    """
    Reads drone coordinates from a file with lines like:
    timestamp: x1, y1, x2, y2, idx, class_name
    Returns a dict: {timestamp: [[x1, y1, x2, y2, idx, class_name], ...]}
    """
    coordinates = {}
    if not os.path.exists(file_path):
        return coordinates # Return empty dict if file doesn't exist
    with open(file_path, "r") as f:
        for line in f:
            try:
                timestamp, rest = line.strip().split(": ")
                parts = rest.split(", ")
                x1, y1, x2, y2, idx = map(float, parts[:5])
                class_name = parts[5]

                if timestamp not in coordinates:
                    coordinates[timestamp] = []
                coordinates[timestamp].append([x1, y1, x2, y2, int(idx), class_name])
            except ValueError:
                # Handle cases where a line might be malformed
                continue
    return coordinates

to_timestamp = lambda idx: str(float("{:.6f}".format((idx+1)*0.033333)))

class FREDDataset(Dataset):
    """
    Corrected Custom PyTorch Dataset for DETR object detection.
    This version provides labels in the COCO-like format expected by DetrImageProcessor.
    """
    def __init__(self, split, dataset_path, class_to_id, modality):
        # The __init__ method remains the same as before.
        self.split = split
        self.dataset_path = dataset_path
        self.class_to_id = class_to_id
        all_paths = get_split_paths(dataset_path, split)
        self.coordinates_cache = {}
        self.all_boxes = {}
        self.valid_image_paths = []
        self.modality = modality
        print(f"Loading '{split}' labels...")
        for rgb_img_path, event_img_path, frame_index in all_paths:
            video_folder = os.path.dirname(event_img_path)
            coordinate_file = video_folder.split('Event')[0] + '/coordinates.txt'
            if coordinate_file not in self.coordinates_cache:
                self.coordinates_cache[coordinate_file] = read_coordinates(coordinate_file)
            coordinates = self.coordinates_cache[coordinate_file]
            timestamp = to_timestamp(frame_index)
            if timestamp in coordinates:
                if modality == 'rgb':
                    self.all_boxes[rgb_img_path] = coordinates[timestamp]
                elif modality == 'event':
                    self.all_boxes[event_img_path] = coordinates[timestamp]
                else:
                    raise ValueError("Modality must be 'rgb' or 'event'")
                self.valid_image_paths.append({'rgb': rgb_img_path, 'event': event_img_path})
        
        # subsample by a factor of 10 for speed
        # if split == 'train':
        #     self.valid_image_paths = self.valid_image_paths[::10]

    def __len__(self):
        return len(self.valid_image_paths)

    def __getitem__(self, idx):
        # Load Image
        img_path = self.valid_image_paths[idx][self.modality]
        image = Image.open(img_path).convert("RGB")

        # Get and Format Labels for the PROCESSOR
        raw_boxes = self.all_boxes.get(img_path, [])
        
        # This is the new part: create a list of COCO-formatted annotations
        coco_annotations = []
        for box_data in raw_boxes:
            x1, y1, x2, y2, _, class_name = box_data

            class_name = 'drone'

            class_id = self.class_to_id.get(class_name)
            if class_id is None:
                continue

            # Convert from [x1, y1, x2, y2] to COCO's [x, y, width, height]
            bbox = [x1, y1, x2 - x1, y2 - y1]
            area = (x2 - x1) * (y2 - y1)
            
            coco_annotations.append({
                "bbox": bbox,
                "category_id": class_id,
                "area": area,
                "iscrowd": 0,
                # The processor needs a unique ID per annotation in the image
                "id": len(coco_annotations), 
            })
        
        # The final target is a dict containing the image_id and the annotations list
        target = {"image_id": idx, "annotations": coco_annotations, 'image_path': img_path}
            
        return image, target
    

class FREDDatasetMultimodal(Dataset):
    def __init__(self, split, dataset_path, class_to_id):
        self.split = split
        self.dataset_path = dataset_path
        self.class_to_id = class_to_id
        self.rgb_dataset = FREDDataset(split, dataset_path, class_to_id, modality='rgb')
        self.event_dataset = FREDDataset(split, dataset_path, class_to_id, modality='event')
        assert len(self.rgb_dataset) == len(self.event_dataset), "RGB and Event datasets must have the same length"
        print(f"Multimodal dataset initialized with {len(self.rgb_dataset)} samples.")

    def __len__(self):
        return min(len(self.rgb_dataset), len(self.event_dataset))
    
    def __getitem__(self, idx):
        rgb_image, _ = self.rgb_dataset[idx]
        event_image, target = self.event_dataset[idx]
        return (rgb_image, event_image), target

if __name__ == "__main__":

    dataset_path = 'FRED/' # if you want to use the canonical dataset
    #dataset_path = 'FRED_challenging/' # if you want to use the challenging dataset
    
    CLASS_TO_ID = {'drone': 0} 
    
    # Instantiate the new dataset
    train_dataset = FREDDatasetMultimodal(
        split='toy', # use the toy split to try it out, use "train" or test "otherwise"
        dataset_path=dataset_path,
        class_to_id=CLASS_TO_ID
    )
    print(f"Number of samples in training set: {len(train_dataset)}")