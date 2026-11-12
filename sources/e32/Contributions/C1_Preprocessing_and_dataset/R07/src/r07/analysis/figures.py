"""Small static scientific SVG figures; no plotting dependency or interpolation."""
from html import escape
from r07.analysis.core import LINES

COLORS = ["#0072B2","#D55E00","#009E73","#CC79A7","#8A6500","#595959"]

def text(x,y,value,size=13,anchor="start"):
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="#17364D">{escape(str(value))}</text>'

def start(width,height,title):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<g font-family="Arial,Helvetica,sans-serif">',text(30,28,title,19)]

def response_figure(rows, fault):
    subset=[r for r in rows if r["fault"]==fault]
    out=start(1160,920,f"Fixed B7B5 response by B7 distance: {fault.upper()}")
    out.append(text(30,51,"Fault-on window RMS relative to paired prefault RMS; points are saved acquisitions, not independent replicates.",12))
    for i,(line,_,_,_) in enumerate(LINES):
        out.append(f'<circle cx="{40+i*181}" cy="76" r="4" fill="{COLORS[i]}"/>')
        out.append(text(50+i*181,81,line.replace("_","-"),12))
    channels=[kind+"B7B5_RMS_"+p for p in "ABC" for kind in ("I","V")]
    for panel,channel in enumerate(channels):
        data=[r for r in subset if r["channel"]==channel]
        x0=76+(panel%2)*565; y0=135+(panel//2)*250; w=470; h=165
        values=[100*r["relative_rms_change"] for r in data]
        low=min(values+[0]);high=max(values+[0]);pad=max((high-low)*.1,.1)
        low-=pad;high+=pad
        out.append(text(x0,y0-14,channel+" change (%)",14))
        out.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#f7f9fb" stroke="#687887"/>')
        for k in range(5):
            y=y0+h-k*h/4;value=low+(high-low)*k/4
            out.append(f'<path d="M{x0} {y}h{w}" stroke="#dce4ea"/>')
            out.append(text(x0-8,y+4,f"{value:.3g}",11,"end"))
        for km in [0,50,100,150,200,250]:
            x=x0+w*km/250
            out.append(text(x,y0+h+18,km,11,"middle"))
        for row in data:
            color=COLORS[[r[0] for r in LINES].index(row["line"])]
            x=x0+w*row["distance_B7_km"]/250
            y=y0+h-h*(100*row["relative_rms_change"]-low)/(high-low)
            tip=f'{row["acquisition_id"]}; local {row["position_km"]} km; B7 {row["distance_B7_km"]} km'
            out.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="4" fill="{color}" fill-opacity=".78"><title>{escape(tip)}</title></circle>')
        out.append(text(x0+w/2,y0+h+39,"Shortest-path distance from B7 (km)",12,"middle"))
    out.append(text(30,904,"Different locations can share a distance. No monotonic attenuation, physical validation or causal distance effect is asserted.",12))
    return "\n".join(out+["</g></svg>"])

def error_figure(rows):
    out=start(1100,470,"Saved logistic-regression errors by fixed B7 distance")
    out.append(text(30,53,"Fault windows only; pooled out-of-fold predictions; proportions are descriptive, not confidence intervals.",12))
    bins=list(dict.fromkeys(r["distance_bin_km"] for r in rows))
    x0=90;y0=100;w=930;h=270
    out.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#f7f9fb" stroke="#687887"/>')
    for k in range(6):
        y=y0+h-k*h/5
        out.append(f'<path d="M{x0} {y}h{w}" stroke="#dce4ea"/>')
        out.append(text(x0-12,y+5,k*20,12,"end"))
    for mi,modality in enumerate(["RMS","RAW","MULTI"]):
        color=COLORS[mi]
        out.append(text(90+mi*300,438,modality+": type errors / fault windows",12))
        out.append(f'<rect x="{70+mi*300}" y="428" width="12" height="12" fill="{color}"/>')
        for bi,b in enumerate(bins):
            row=next(r for r in rows if r["modality"]==modality and r["distance_bin_km"]==b)
            count=row["fault_windows"];rate=row["type_error_fraction"]
            x=x0+(bi+.5)*w/len(bins)+(mi-1)*24
            if rate is not None:
                height=h*rate
                out.append(f'<rect x="{x-9}" y="{y0+h-height}" width="18" height="{height}" fill="{color}"/>')
            if mi==1:
                out.append(text(x0+(bi+.5)*w/len(bins),y0+h+23,b,12,"middle"))
                out.append(text(x0+(bi+.5)*w/len(bins),y0+h+43,f"n={count} fault windows/profile",11,"middle"))
    return "\n".join(out+["</g></svg>"])

def html_report(summary, fault_labels, audit_rows):
    links="\n".join(f'<h2>{escape(f.upper())}</h2><img src="figures/response_{escape(f)}.svg" alt="{escape(f)} response">' for f in fault_labels)
    issues=sum(len(r["issues"]) for r in audit_rows)
    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>R07 saved distance analysis</title><style>
body{{font:16px/1.5 Arial,sans-serif;color:#17364D;max-width:1160px;margin:30px auto;padding:0 20px}}
img{{max-width:100%;height:auto}}table{{border-collapse:collapse}}td,th{{padding:7px;border:1px solid #ccd6df}}code{{overflow-wrap:anywhere}}
</style><h1>R07 saved distance analysis</h1>
<p>Classifier: standardized multinomial logistic regression. Three fixed descriptor profiles; five acquisition-grouped, fault-type-stratified folds, seed 42. Approximately 80% train/20% held-out per fold. No separate tuning-validation set, hyperparameter search or final holdout. Scaling was learned only on training folds.</p>
<p>Data: 429 fault acquisitions, each with paired prefault and fault windows (858 total). This analysis compares the same fault type across its 39 spatial locations, always measured at B7B5. No added noise or denoising; no new fits or simulations. Recovery samples are not required or used.</p>
<p>RMS effects are (fault RMS - prefault RMS)/prefault RMS. The two 200 ms windows include their recorded transients. This is a window summary, not instantaneous waveform or recovery analysis.</p>
<h2>Interpretation</h2><p>Local line position differs from shortest-path distance to B7. Several locations can share a B7 distance. The six lines have unequal numbers of acquisitions. Markers are not connected: distance does not define one ordered physical trajectory. Impedance, fault type, power flow and alternate paths can affect response. Do not infer causal attenuation or unseen-location generalization.</p>
<h2>Saved classification errors</h2><img src="figures/classification_errors.svg" alt="Fault-type errors by B7 distance">
<p>Denominators exclude prefault windows: noFault is not assigned the future fault's distance. A type error includes a missed fault. Distance-bin counts remain visible; empty bins are unavailable, not zero-error results. Full per-type/per-line tables are included.</p>
<h2>ABC versus ABCG audit</h2><p>Paired saved configuration/build review: {len(audit_rows)} matched locations; {issues} issue records. Read <a href="abc_abcg_configuration_audit.json">the exact findings</a>. Matched flags and masks do not certify fault physics, ground-path impedance or actual contact opening. The scheduled command is not independently observed contact telemetry.</p>
<p>The fourteen E16 voltage discrepancies and model/segmentation/bandwidth limitations remain open. A persistent fault may have no recovery phase. This fixed-duration dataset does not validate persistent-fault or variable-duration generalization.</p>
<h2>Tables and provenance</h2><p><a href="signal_effects.csv">Signal effects</a> |
<a href="classification_by_distance.csv">Distance-bin errors</a> |
<a href="classification_by_type_line.csv">Type/line errors</a> |
<a href="classification_by_type_distance.csv">Type/distance errors</a> |
<a href="case_distances.csv">Spatial coordinates</a> |
<a href="analysis_report.json">Summary and hashes</a></p>{links}</html>"""
