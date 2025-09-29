from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class CsvRecord:
    image_path: str
    text: str


class ImageTextCsvDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        multi_view_transforms: List[Callable],
        image_backend: str = "pil",
    ) -> None:
        self.csv_path = csv_path
        self.records: List[CsvRecord] = self._load_csv(csv_path)
        self.multi_view_transforms = multi_view_transforms
        self.image_backend = image_backend

    def _load_csv(self, csv_path: str) -> List[CsvRecord]:
        df = pd.read_csv(csv_path)
        if not {"image_path", "text"}.issubset(set(df.columns)):
            raise ValueError("CSV must contain columns: image_path,text")
        records: List[CsvRecord] = []
        for _, row in df.iterrows():
            image_path = str(row["image_path"]).strip()
            text = str(row["text"]).strip()
            if not os.path.isabs(image_path):
                image_path = os.path.abspath(image_path)
            records.append(CsvRecord(image_path=image_path, text=text))
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, path: str) -> Image.Image:
        with Image.open(path) as img:
            return img.convert("RGB")

    def __getitem__(self, index: int):
        
        rec = self.records[index]
        image = self._load_image(rec.image_path)
        image_views = [t(image) for t in self.multi_view_transforms]
        return image_views, rec.text #(bsz,views,C,H,W)