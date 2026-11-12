"""Column-width layout of the existing C1 network, with verified model mapping.

Run from PROJECT_ROOT; this changes only the figure layout and typography.
PDF and SVG share the same drawing commands. Rasterize the PDF for visual QA.
"""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('manuscript/revision_complete_20261028/evidence/parent_static_model_config.json'))
    parser.add_argument('--output', type=Path, default=Path('manuscript/Contrib1_P1_Preprocessing_paper/Figures/network_model_column.pdf'))
    parser.add_argument('--font-regular', type=Path, default=Path('C:/Windows/Fonts/arial.ttf'))
    parser.add_argument('--font-bold', type=Path, default=Path('C:/Windows/Fonts/arialbd.ttf'))
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes.decode('utf-8-sig'))
    loads = {int(bus): (p, q) for bus, p, q in config['paper_load_bus_p_mw_q_mvar']}
    assert loads == {5:(125,50), 6:(90,30), 8:(100,35)}
    assert config['paper_auxiliary_load_bus'] == 6
    assert (config['auxiliary_load']['p_mw'], config['auxiliary_load']['q_mvar']) == (50,0)

    ring = [('B7','B8'),('B8','B9'),('B9','B6'),('B6','B4'),('B4','B5'),('B5','B7')]
    couplings = [('B1','B4'),('B2','B7'),('B3','B9')]
    inverse = config['internal_to_paper_bus']
    model_edges = {frozenset((f'B{inverse[int(a)-1]}',f'B{inverse[int(b)-1]}'))
                   for a,b,*_ in config['branches_from_to_r_x_b_pu']}
    assert model_edges == {frozenset(e) for e in ring+couplings}
    assert len(model_edges) == 9

    pdfmetrics.registerFont(TTFont('C1Arial', str(args.font_regular)))
    pdfmetrics.registerFont(TTFont('C1ArialBold', str(args.font_bold)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    width, height = 246, 258
    c = Canvas(str(args.output),pagesize=(width,height),invariant=1,pageCompression=1)
    c.setTitle('Simulated Nine-Bus Network and Measurement Point')
    c.setAuthor('C1 manuscript; AUD16 layout revision')
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}pt" height="{height}pt" viewBox="0 0 {width} {height}">',
           '<title>Simulated Nine-Bus Network and Measurement Point</title>',
           '<rect width="100%" height="100%" fill="white"/>']
    labels=[]
    dark='#1f3543'; branch='#225772'; coupling='#526976'; measure='#007a86'

    def color(value):
        return tuple(int(value[i:i+2],16)/255 for i in (1,3,5))

    def line(x1,y1,x2,y2,col=dark,lw=0.8,dash=None):
        c.setStrokeColorRGB(*color(col));c.setLineWidth(lw);c.setDash(dash or [])
        c.line(x1,y1,x2,y2);c.setDash([])
        extra=f' stroke-dasharray="{",".join(str(x) for x in dash)}"' if dash else ''
        svg.append(f'<line x1="{x1}" y1="{height-y1}" x2="{x2}" y2="{height-y2}" stroke="{col}" stroke-width="{lw}"{extra}/>')

    def circle(x,y,r,fill=True):
        c.setStrokeColorRGB(*color(dark));c.setFillColorRGB(1,1,1);c.setLineWidth(0.8)
        c.circle(x,y,r,stroke=1,fill=int(fill))
        svg.append(f'<circle cx="{x}" cy="{height-y}" r="{r}" fill="{"white" if fill else "none"}" stroke="{dark}" stroke-width="0.8"/>')

    def polygon(points,col):
        c.setFillColorRGB(*color(col));p=c.beginPath();p.moveTo(*points[0])
        for xy in points[1:]:p.lineTo(*xy)
        p.close();c.drawPath(p,stroke=0,fill=1)
        pts=' '.join(f'{x},{height-y}' for x,y in points)
        svg.append(f'<polygon points="{pts}" fill="{col}"/>')

    def arrow(x1,y1,x2,y2,col=dark,lw=0.8):
        line(x1,y1,x2,y2,col,lw)
        ang=math.atan2(y2-y1,x2-x1)
        polygon([(x2,y2),(x2-3.5*math.cos(ang-0.4),y2-3.5*math.sin(ang-0.4)),
                 (x2-3.5*math.cos(ang+0.4),y2-3.5*math.sin(ang+0.4))],col)

    def text(x,y,value,size=8.2,col=dark,bold=False,align='center'):
        font='C1ArialBold' if bold else 'C1Arial'
        tw=pdfmetrics.stringWidth(value,font,size)
        x0=x-tw/2 if align=='center' else x
        asc=pdfmetrics.getAscent(font)*size/1000
        desc=pdfmetrics.getDescent(font)*size/1000
        box=[x0,y+desc,x0+tw,y+asc]
        assert box[0]>=1 and box[2]<=width-1 and box[1]>=1 and box[3]<=height-1,(value,box)
        c.setFillColorRGB(*color(col));c.setFont(font,size);c.drawString(x0,y,value)
        anchor='middle' if align=='center' else 'start'
        weight='bold' if bold else 'normal'
        svg.append(f'<text x="{x}" y="{height-y}" font-family="Arial,sans-serif" font-size="{size}" font-weight="{weight}" fill="{col}" text-anchor="{anchor}">{html.escape(value)}</text>')
        labels.append({'text':value,'size_pt':size,'bbox_pdf_pt':box})

    nodes={'B7':(77,177),'B8':(163,177),'B9':(195,119),
           'B6':(163,61),'B4':(77,61),'B5':(45,119),
           'B1':(18,61),'B2':(20,207),'B3':(229,119)}
    for a,b in ring:line(*nodes[a],*nodes[b],branch,1.1)
    for a,b in couplings:line(*nodes[a],*nodes[b],coupling,0.8,[3,2])
    for name,(x,y) in nodes.items():
        if name in {'B1','B2','B3'}:
            circle(x,y,9.2);circle(x,y,7.1,fill=False)
        else:circle(x,y,7.5)
        text(x,y-2.6,name,8.2)
    text(24,42,'Fixed source',7.8)
    text(25,222,'Fixed source',7.8)
    text(221,102,'Fixed source',7.8)
    text(width/2,246,'Simulated Nine-Bus Network and Measurement Point',9.0,bold=True)
    text(width/2,234,'Load ratings: nominal MW / MVAr',8.0)

    arrow(163,185,163,204)
    text(163,222,'Variable B8',8.2)
    text(163,211,'100 / 35',8.2)
    arrow(37,119,8,119)
    text(29,156,'Variable B5',8.2)
    text(29,145,'125 / 50',8.2)
    line(163,53,163,43);line(126,43,202,43)
    arrow(126,43,126,33);arrow(202,43,202,33)
    text(126,22,'Variable B6',8.2)
    text(126,11,'90 / 30',8.2)
    text(202,22,'Fixed auxiliary',8.2)
    text(202,11,'50 / 0',8.2)

    # Marker is near B7 on B7--B5, not another bus or measurement point.
    fraction=0.22
    mx=77+fraction*(45-77);my=177+fraction*(119-177)
    arrow(90,161,mx,my,measure,0.9)
    line(mx-2,my-2,mx+2,my-2,measure,1.4)
    line(mx+2,my-2,mx+2,my+2,measure,1.4)
    line(mx+2,my+2,mx-2,my+2,measure,1.4)
    line(mx-2,my+2,mx-2,my-2,measure,1.4)
    text(120,156,'Fixed I, V',8.6,col=measure,bold=True)
    text(120,144,'B7 end of B7-B5',8.2,col=measure)
    line(83,109,98,109,branch,1.1)
    text(104,106,'Fault-location branches',7.8,align='left')
    line(83,94,98,94,coupling,0.8,[3,2])
    text(104,91,'Source couplings',7.8,align='left')

    overlaps=[]
    for i,a in enumerate(labels):
        for b in labels[i+1:]:
            aa,bb=a['bbox_pdf_pt'],b['bbox_pdf_pt']
            if min(aa[2],bb[2])-max(aa[0],bb[0])>0.2 and min(aa[3],bb[3])-max(aa[1],bb[1])>0.2:
                overlaps.append((a['text'],b['text']))
    assert not overlaps,overlaps
    c.showPage();c.save();svg.append('</svg>')
    args.output.with_suffix('.svg').write_text('\n'.join(svg)+'\n',encoding='utf-8')
    audit={'config_path':args.config.as_posix(),'config_sha256':hashlib.sha256(config_bytes).hexdigest(),
           'page_pt':[width,height],'minimum_font_size_pt':min(t['size_pt'] for t in labels),
           'text_bbox_overlap_count':len(overlaps),'nodes':nodes,'ring_edges':ring,
           'source_couplings':couplings,'loads':loads,'auxiliary_load':{'bus':6,'P_MW':50,'Q_MVAr':0},
           'measurement':{'branch':['B7','B5'],'terminal':'B7','channels':'three I and three V'},
           'model_edges_verified':True,'topology_or_physical_values_changed':False,'labels':labels}
    args.output.with_suffix('.layout.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(args.output),'page_pt':[width,height],
                      'minimum_font_pt':audit['minimum_font_size_pt'],'overlaps':len(overlaps),'model_edges_verified':True}))

if __name__=='__main__':main()
