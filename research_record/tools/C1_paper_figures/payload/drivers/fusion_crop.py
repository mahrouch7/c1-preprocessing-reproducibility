"""Retain complete SNR curves; table 2 carries original inferential outcomes.

Vector-only crop: no axis or score changed. Removes the repeated 10-dB atlas.
"""
from pathlib import Path
import io
from reportlab.pdfgen.canvas import Canvas
from pypdf import PdfReader,PdfWriter,Transformation
P=Path(__file__).resolve().parents[1]
ROOT=P.parents[1]
src=P/'intermediate/fusion_titled.pdf'
page=PdfReader(src).pages[0]
page.mediabox.lower_left=(0,118);page.cropbox.lower_left=(0,118)
page.mediabox.upper_right=(522,294);page.cropbox.upper_right=(522,294)
buf=io.BytesIO();c=Canvas(buf,pagesize=(522,190),invariant=1)
c.setFont('Helvetica',8);c.drawCentredString(261,3,'Additive SNR (dB); positive changes favour fusion over the named constituent')
c.showPage();c.save()
out=PdfReader(io.BytesIO(buf.getvalue())).pages[0]
out.merge_transformed_page(page,Transformation().translate(0,-104))
w=PdfWriter();w.add_page(out)
with (P/'intermediate/fusion_final.pdf').open('wb') as f:w.write(f)
print('Fusion curves exported; table 2 retains all 160 original fusion contrasts.')
