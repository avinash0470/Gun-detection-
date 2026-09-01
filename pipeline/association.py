from typing import Dict, List, Tuple
from .tracking import TrackedPerson

class PersonGunAssociation:
    def __init__(self):
        pass

    def associate(self, 
                  tracked_persons: Dict[str, TrackedPerson], 
                  detected_guns: List[Dict]) -> List[Tuple[str, Dict]]:
        """
        Associates detected firearms with specific tracked persons.
        Checks for bounding box containment (gun inside person box) & spatial proximity.
        Returns:
            List[Tuple[str, Dict]]: List of (track_id, gun_detection_dict)
        """
        associations = []

        for gun in detected_guns:
            gun_bbox = gun["bbox"]
            gx1, gy1, gx2, gy2 = gun_bbox
            g_center = ((gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0)

            best_match_id = None
            highest_score = 0.0

            for pid, person in tracked_persons.items():
                px1, py1, px2, py2 = person.bbox
                
                # 1. Containment Check: Is gun box center inside person bounding box?
                is_center_contained = (px1 <= g_center[0] <= px2) and (py1 <= g_center[1] <= py2)
                
                # 2. Compute Intersection Over Gun Area (IoGA)
                inter_x1 = max(px1, gx1)
                inter_y1 = max(py1, gy1)
                inter_x2 = min(px2, gx2)
                inter_y2 = min(py2, gy2)
                
                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                gun_area = float((gx2 - gx1) * (gy2 - gy1) + 1e-6)
                overlap_ratio = inter_area / gun_area # Fraction of gun box inside person box

                # 3. Spatial center distance score
                p_center = ((px1 + px2) / 2.0, (py1 + py2) / 2.0)
                dist = ((p_center[0] - g_center[0])**2 + (p_center[1] - g_center[1])**2)**0.5
                
                score = 0.0
                if is_center_contained:
                    score += 0.6
                score += (overlap_ratio * 0.4)

                # Accept association if gun is inside/overlapping or within reasonable spatial distance (< 250px)
                if (score > 0.3 or dist < 250) and score >= highest_score:
                    highest_score = score
                    best_match_id = pid

            if best_match_id:
                associations.append((best_match_id, gun))
                tracked_persons[best_match_id].association_history.append(gun.get("detector", "unknown"))

        return associations
