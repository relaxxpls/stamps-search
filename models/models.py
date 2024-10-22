from dataclasses import dataclass
from typing import List, Dict


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    @property
    def area(self) -> float:
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)

    @classmethod
    def calc_overlap(cls, box1: "BoundingBox", box2: "BoundingBox") -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        """
        x1 = max(box1.xmin, box2.xmin)
        y1 = max(box1.ymin, box2.ymin)
        x2 = min(box1.xmax, box2.xmax)
        y2 = min(box1.ymax, box2.ymax)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        smaller_area = min(box1.area, box2.area)
        if smaller_area == 0:
            return 0

        return intersection / smaller_area


@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> "DetectionResult":
        box = BoundingBox(
            xmin=detection_dict["box"]["xmin"],
            ymin=detection_dict["box"]["ymin"],
            xmax=detection_dict["box"]["xmax"],
            ymax=detection_dict["box"]["ymax"],
        )

        return cls(
            score=detection_dict["score"], label=detection_dict["label"], box=box
        )
