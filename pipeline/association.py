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
        Returns:
            List[Tuple[str, Dict]]: List of (track_id, gun_detection_dict)
        """
        associations = []

        # Loop through detected guns and map to the closest tracked person
        for gun in detected_guns:
            gun_bbox = gun["bbox"]
            best_match_id = None
            min_dist = float("inf")

            for pid, person in tracked_persons.items():
                p_bbox = person.bbox
                # Spatial distance metric: distance between centers
                p_center = ((p_bbox[0] + p_bbox[2]) / 2, (p_bbox[1] + p_bbox[3]) / 2)
                g_center = ((gun_bbox[0] + gun_bbox[2]) / 2, (gun_bbox[1] + gun_bbox[3]) / 2)

                dist = ((p_center[0] - g_center[0])**2 + (p_center[1] - g_center[1])**2)**0.5
                
                # In production, we also check ReID trajectory history & hand overlap
                # to prevent wrong assignments.
                # Threshold of spatial proximity:
                if dist < 200 and dist < min_dist:
                    min_dist = dist
                    best_match_id = pid

            if best_match_id:
                associations.append((best_match_id, gun))
                # Update tracking history
                tracked_persons[best_match_id].association_history.append(gun.get("detector", "unknown"))

        return associations
