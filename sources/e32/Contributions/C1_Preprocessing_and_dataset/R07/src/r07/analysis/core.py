"""Pure spatial and keyed-table contracts. No learning or simulator imports."""
import heapq
import math

LINES = [("B7_B8",7,8,50),("B8_B9",8,9,100),("B9_B6",9,6,100),
         ("B6_B4",6,4,50),("B4_B5",4,5,50),("B5_B7",5,7,100)]
BINS = (0,50,100,150,200,250)

def require(condition, message):
    if not condition:
        raise ValueError(message)

def unique(rows, key):
    result = {}
    for row in rows:
        value = row[key]
        require(value not in result, "Duplicate key: "+str(value))
        result[value] = row
    return result

def distances(reference=7, lines=LINES):
    adjacency = {}
    for _,u,v,length in lines:
        require(math.isfinite(length) and length>0, "Invalid graph length")
        adjacency.setdefault(u,[]).append((v,length))
        adjacency.setdefault(v,[]).append((u,length))
    require(reference in adjacency, "Reference missing from graph")
    result = {reference:0}
    queue = [(0,reference)]
    while queue:
        distance,u = heapq.heappop(queue)
        if distance != result[u]:
            continue
        for v,length in adjacency[u]:
            if distance+length < result.get(v,math.inf):
                result[v] = distance+length
                heapq.heappush(queue,(distance+length,v))
    require(len(result)==len(adjacency), "Disconnected distance graph")
    return result

def fault_distance(line, position, reference=7):
    matches = [r for r in LINES if r[0]==line]
    require(len(matches)==1, "Unknown line")
    _,u,v,length = matches[0]
    require(not isinstance(position,bool) and math.isfinite(position) and 0<position<length,
            "Fault position outside line interior")
    path = distances(reference)
    return min(path[u]+position, path[v]+length-position)

def distance_bin(distance, edges=BINS):
    require(len(edges)>1 and all(b>a for a,b in zip(edges,edges[1:])), "Invalid bin edges")
    require(math.isfinite(distance), "Nonfinite distance")
    for a,b in zip(edges,edges[1:]):
        if a<=distance<b:
            return f"[{a},{b})"
    raise ValueError("Distance outside declared bins")

def relative_change(before, during):
    require(math.isfinite(before) and math.isfinite(during) and before>0 and during>=0,
            "Invalid RMS or zero prefault denominator; no silent imputation")
    value=(during-before)/before
    require(math.isfinite(value), 'RMS ratio overflow; no silent infinity')
    return value

def differences(left, right, prefix=""):
    """Exact JSON-tree differences; handles missing keys rather than filling them."""
    if isinstance(left,dict) and isinstance(right,dict):
        out = []
        for key in sorted(set(left)|set(right)):
            path = prefix+"."+str(key) if prefix else str(key)
            if key not in left or key not in right:
                out.append(path)
            else:
                out.extend(differences(left[key],right[key],path))
        return out
    if isinstance(left,list) and isinstance(right,list):
        if len(left)!=len(right):
            return [prefix]
        out = []
        for i,(a,b) in enumerate(zip(left,right)):
            out.extend(differences(a,b,f"{prefix}[{i}]"))
        return out
    return [] if type(left)==type(right) and left==right else [prefix]
