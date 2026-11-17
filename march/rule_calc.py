class CSI:
    HIGH_RISK = "High risk"
    MODERATE_RISK = "Moderate risk"
    LOW_RISK = "Low risk"

class TBI:
    HIGH_RISK = "High risk"
    LOW_RISK = "Low risk"

def run_cspine_rule(row):
    if row.risk_gcs <= 8: return CSI.HIGH_RISK
    if str(row.risk_mentalstatus) == "Unresponsive": return CSI.HIGH_RISK
    if row.risk_abc == "Yes": return CSI.HIGH_RISK
    if row.risk_focal == "Yes": return CSI.HIGH_RISK

    if str(row.risk_mentalstatus) != "Alert": return CSI.MODERATE_RISK
    if row.risk_gcs < 15: return CSI.MODERATE_RISK
    if row.risk_tenderness == "Yes" or row.risk_pain == "Yes": return CSI.MODERATE_RISK
    if row.risk_torsoinj == "Yes" or row.risk_headinj == "Yes": return CSI.MODERATE_RISK

    assert row.risk_gcs == 15 and row.risk_abc == "No" and row.risk_mentalstatus == "Alert" and row.risk_pain == "No" and row.risk_headinj == "No" and row.risk_focal == "No" and row.risk_tenderness == "No" and row.risk_torsoinj == "No"

    return CSI.LOW_RISK

def run_tbi_younger_rule(row):
    risks_without_baseline = ["risk_loc", "risk_skullfx", "risk_altered", "risk_hematoma", "risk_mechanism"]
    for r in risks_without_baseline:
        if row[r] == "Yes": return TBI.HIGH_RISK
    
    if row["risk_baseline"] == "No":
        return TBI.HIGH_RISK
    
    assert all([row[r] == "No" for r in risks_without_baseline]) and row["risk_baseline"] == "Yes"
    return TBI.LOW_RISK

def run_tbi_older_rule(row):
    try:
        risks = ["risk_mechanism", "risk_headache", "risk_vomit", "risk_loc", "risk_sf", "risk_altered"]
        
        if any([row[r] == "Yes" for r in risks]):
            return TBI.HIGH_RISK
        
        assert all([row[r] == "No" for r in risks])
        return TBI.LOW_RISK
    except Exception as e:
        print(row)
        print(e)

# IAI["risk_altered"] = IAI.risk_gcs.apply(lambda gcs: "Yes" if gcs < 14 else "No")
# IAI["rule"] = IAI.apply(run_iai_rule, axis=1)

def run_iai_rule(row):
    risk_altered = "Yes" if row.risk_gcs < 14 else "No"
    try:
        risks = ["risk_vomit", "risk_breathing", "risk_abdpain", "risk_walltrauma", "risk_chestwall", "risk_tenderness"]
        risks = [row[r] for r in risks] + [risk_altered]
        
        if any([r == "Yes" for r in risks]):
            return TBI.HIGH_RISK
        
        assert all([r == "No" for r in risks])
        return TBI.LOW_RISK
    except Exception as e:
        print(row)
        print(e)
