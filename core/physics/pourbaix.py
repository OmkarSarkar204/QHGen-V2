class PourbaixFilter:
    def __init__(self):
        self.stability_data = {
            'Ni': {'min': 8, 'max': 14},
            'Fe': {'min': 9, 'max': 14},
            'Co': {'min': 7, 'max': 14},
            'Cu': {'min': 6, 'max': 13},
            'Pt': {'min': 0, 'max': 14},
            'Au': {'min': 0, 'max': 14},
            'Ti': {'min': 2, 'max': 12},
            'Mo': {'min': 4, 'max': 14},
            'W':  {'min': 4, 'max': 11},
            'Al': {'min': 4, 'max': 8.5},
            'Zn': {'min': 9, 'max': 11}
        }

    def check_stability(self, elements, ph):
        rejected = []
        for el in elements:
            if el in self.stability_data:
                data = self.stability_data[el]
                if ph < data['min'] or ph > data['max']:
                    rejected.append(f"{el} (Stable: {data['min']}-{data['max']})")
        
        if rejected:
            return False, rejected
        return True, []