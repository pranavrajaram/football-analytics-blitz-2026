def simplify_route(r):
    if not r: return 'No Route'
    r = r.lower()
    if 'no route' in r or 'block' in r: return 'No Route'
    if 'post' in r: return 'Post'
    if 'corner' in r: return 'Corner'
    if 'seam' in r or 'streak' in r or 'go' in r or 'fly' in r or 'vertical' in r: return 'Vertical'
    if 'out' in r: return 'Out'
    if 'dig' in r or 'in' in r: return 'Cross'
    if 'drag' in r or 'shallow' in r: return 'Shallow'
    if 'cross' in r or 'over' in r: return 'Cross'
    if 'screen' in r: return 'Screen'
    if 'flat' in r: return 'Flat'
    return 'No Route'

routes = {
  "L1": "Vertical",
  "TE": "Post",
  "R1": "Corner",
  "R2": "Cross",
  "R3": "Flat",
  "L2": "Post"
}

print(f"Original: {routes}")

target_slots = ['R2', 'L2', 'R3', 'L3', 'R4', 'L4']
legacy_keys = [k for k in routes.keys() if k not in ['L1', 'L2', 'L3', 'L4', 'R1', 'R2', 'R3', 'R4']]

for k in legacy_keys:
    assigned = False
    for slot in target_slots:
        if slot not in routes:
            routes[slot] = routes[k]
            assigned = True
            break
    if not assigned:
        for slot in ['R1', 'L1']:
            if slot not in routes:
                routes[slot] = routes[k]
                break

print(f"Mapped: {routes}")
print(f"L3 Value: {routes.get('L3')}")
print(f"L3 Simplified: {simplify_route(routes.get('L3'))}")
