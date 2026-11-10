
def grade_observation(parsed):
    if not isinstance(parsed, dict):
        return None
    assessment = parsed.get("assessment")
    evidence = parsed.get("evidence")
    if assessment == "no_visible_impact" and evidence == []:
        return 0
    if assessment not in ("multiple_sites", "route_disrupted", "localized") or not isinstance(evidence,list):
        return None
    regions = {"upper left","upper center","upper right","center left","center","center right","lower left","lower center","lower right"}
    locations = set()
    for row in evidence:
        if not isinstance(row,dict) or any(not isinstance(row.get(k),str) or not row[k].strip() for k in ("what","where","sign")):
            return None
        where = row["where"].lower().strip().replace("-"," ").removesuffix(" region").removesuffix(" quadrant")
        if where not in regions:
            return None
        locations.add(where)
    if assessment == "multiple_sites":
        return 2 if len(locations) >= 2 else None
    return (2 if assessment == "route_disrupted" else 1) if locations else None
