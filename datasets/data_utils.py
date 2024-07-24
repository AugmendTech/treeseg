

def to_hhmmss(x,include_milli=False):
    hh = int(x // 3600)
    mm = int((x - 3600 * hh) // 60)

    if not include_milli:
        ss = int((x - 3600 * hh - 60 * mm) // 1)
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    
    ss = (x - 3600 * hh - 60 * mm)
    return f"{hh:02d}:{mm:02d}:{ss:06.3f}"
    

def strip_key(s):
    return s.split("id(")[-1].split(")")[0]   


class AnnoTreeNode:
    def __init__(self,):
        self.nn = []
        self.composed = False
        self.keys = []
        self.convo = []
