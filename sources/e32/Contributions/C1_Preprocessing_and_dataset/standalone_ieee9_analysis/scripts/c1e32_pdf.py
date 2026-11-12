"""Strict generated-SVG to vector-PDF export, using the existing document runtime.

No scripts, external SVG resources, bitmap embedding or dependency installation.
This module intentionally does not import the scientific environment.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

DEFAULT_PYTHON = r'C:\Users\Maher369\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
DEFAULT_FONT = r'C:\Windows\Fonts\arial.ttf'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def runtime(font):
    import reportlab
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    font=Path(font)
    if not font.is_file(): raise ValueError('Existing embedding font missing; no installation or substitution')
    pdfmetrics.registerFont(TTFont('C1Publication',str(font)))
    return {'reportlab':reportlab.Version,'font_sha256':sha(font),'python':__import__('platform').python_version(),
        'font_name':'C1Publication','font_source':str(font),'python_executable':os.path.realpath(__import__('sys').executable)}

def color(value):
    from reportlab.lib.colors import toColor, Color
    if value=='none': return None
    if value.startswith('rgb(') and value.endswith(')'):
        channels=[float(v)/255 for v in value[4:-1].split(',')]
        if len(channels)!=3 or any(not 0<=v<=1 for v in channels): raise ValueError('Invalid RGB')
        return Color(*channels)
    return toColor(value)

def convert(source,destination):
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    source,destination=Path(source),Path(destination)
    if destination.exists(): raise ValueError('PDF output already exists: '+str(destination))
    root=ET.parse(source).getroot(); width=float(root.attrib['width']); height=float(root.attrib['height'])
    if root.attrib.get('viewBox')!=f'0 0 {width:g} {height:g}': raise ValueError('Unexpected SVG coordinate system')
    texts=[float(e.attrib['font-size']) for e in root.iter() if e.tag.endswith('}text')]
    minimum=min(texts) if texts else 12
    publication=source.stem.startswith('publication_')
    # Review atlases remain large enough to inspect. Dedicated panels are 7 inches wide.
    page_width=504 if publication else max(504,width*8/minimum)
    scale=page_width/width; page_height=height*scale
    if publication and minimum*scale<8: raise ValueError('Publication font would be smaller than 8 pt')
    pdf=canvas.Canvas(str(destination),pagesize=(page_width,page_height),pageCompression=1,invariant=1)
    pdf.setTitle(source.stem); pdf.setAuthor('C1 scientific analysis'); pdf.setSubject('Vector scientific figure; fixed family-held-out protocol')
    def number(value,reference): return reference if value=='100%' else float(value)
    def draw(element,inherited):
        tag=element.tag.rsplit('}',1)[-1]; attrs={**inherited,**element.attrib}
        if any(k in attrs for k in ('transform','href','style')): raise ValueError('Unsupported SVG attribute')
        if tag in ('svg','g'):
            for child in element: draw(child,attrs)
            return
        fill=color(attrs.get('fill','#172b3a')); stroke=color(attrs.get('stroke','none'))
        pdf.setLineWidth(float(attrs.get('stroke-width',1))*scale)
        pdf.setDash([float(v)*scale for v in attrs.get('stroke-dasharray','').split(',') if v] or [])
        if fill: pdf.setFillColor(fill)
        if stroke: pdf.setStrokeColor(stroke)
        if tag=='rect':
            x=float(attrs.get('x',0)); y=float(attrs.get('y',0))
            w=number(attrs['width'],width); h=number(attrs['height'],height)
            pdf.rect(x*scale,(height-y-h)*scale,w*scale,h*scale,stroke=int(stroke is not None),fill=int(fill is not None))
        elif tag=='circle':
            pdf.circle(float(attrs['cx'])*scale,(height-float(attrs['cy']))*scale,float(attrs['r'])*scale,
                       stroke=int(stroke is not None),fill=int(fill is not None))
        elif tag=='polyline':
            points=[tuple(map(float,p.split(','))) for p in attrs['points'].split()]
            path=pdf.beginPath()
            for i,(x,y) in enumerate(points):
                if not math.isfinite(x+y): raise ValueError('Nonfinite SVG coordinates')
                (path.moveTo if i==0 else path.lineTo)(x*scale,(height-y)*scale)
            pdf.drawPath(path,stroke=int(stroke is not None),fill=int(fill is not None))
        elif tag=='text':
            value=''.join(element.itertext()); size=float(attrs['font-size'])*scale
            x=float(attrs['x'])*scale; y=(height-float(attrs['y']))*scale
            text_width=pdfmetrics.stringWidth(value,'C1Publication',size)
            anchor=attrs.get('text-anchor','start')
            if anchor=='middle': x-=text_width/2
            elif anchor=='end': x-=text_width
            elif anchor!='start': raise ValueError('Unknown text anchor')
            if x<-.5 or x+text_width>page_width+.5: raise ValueError('Text exceeds figure width: '+source.name+' / '+value)
            pdf.setFont('C1Publication',size); pdf.drawString(x,y,value)
        else: raise ValueError('Unsupported generated SVG element: '+tag)
    draw(root,{})
    pdf.showPage(); pdf.save()
    return {'svg':source.name,'svg_sha256':sha(source),'pdf':destination.name,'pdf_sha256':sha(destination),
        'page_width_pt':round(page_width,3),'page_height_pt':round(page_height,3),
        'minimum_font_pt':round(minimum*scale,3),'embedded_font':True,'vector_only':True,
        'purpose':'TWO_COLUMN_WIDTH_CANDIDATE_VISUAL_REVIEW_REQUIRED' if publication else 'LARGE_REVIEW_ATLAS_USE_PUBLICATION_PANELS_FOR_MANUSCRIPT'}

def export(directory,font):
    directory=Path(directory); evidence=runtime(font); target=directory/'pdf'; target.mkdir()
    rows=[]
    for source in sorted(directory.glob('*.svg')):
        rows.append(convert(source,target/(source.stem+'.pdf')))
    if not rows: raise ValueError('No SVG figures to convert')
    with (directory/'pdf_manifest.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    evidence.update(status='VECTOR_PDF_EXPORT_COMPLETE_VISUAL_REVIEW_REMAINS',figures=len(rows),
        publication_candidates=sum(r['svg'].startswith('publication_') for r in rows))
    (directory/'pdf_export_receipt.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
    lines=['% Copy the chosen PDF into your Overleaf figures/ folder. Do not import the entire atlas.',
           '% Two-column examples use figure* and width=\\textwidth; review labels before inclusion.']
    for row in rows:
        if row['svg'].startswith('publication_'):
            lines.append('% \\includegraphics[width=\\textwidth]{figures/'+row['pdf']+'}')
    (directory/'overleaf_include_examples.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('[VECTOR PDF COMPLETE] '+str(len(rows))+' figures; '+str(evidence['publication_candidates'])+' two-column candidates',flush=True)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--directory',type=Path); parser.add_argument('--font',default=DEFAULT_FONT)
    parser.add_argument('--check',action='store_true'); args=parser.parse_args()
    if args.check: print(json.dumps(runtime(args.font),sort_keys=True)); return
    if args.directory is None: parser.error('--directory is required for export')
    export(args.directory,args.font)

if __name__=='__main__': main()
