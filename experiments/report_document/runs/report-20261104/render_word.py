from pathlib import Path
import win32com.client
import fitz
from PIL import Image,ImageDraw
r=Path('experiments/report_document/runs/report-20261104').resolve()
w=win32com.client.DispatchEx('Word.Application');w.Visible=False
try:
 d=w.Documents.Open(str(r/'Semantic_receiver_scheduler_report.docx'));d.ExportAsFixedFormat(str(r/'qa.pdf'),17);d.Close(False)
finally:w.Quit()
f=fitz.open(r/'qa.pdf');print('PAGES',len(f))
thumbs=[]
for i,page in enumerate(f):
 pix=page.get_pixmap(matrix=fitz.Matrix(1,1));pix.save(r/f'page-{i+1}.png')
 im=Image.open(r/f'page-{i+1}.png').convert('RGB');im.thumbnail((255,340));canvas=Image.new('RGB',(275,365),'#cccccc');canvas.paste(im,(10,20));ImageDraw.Draw(canvas).text((10,3),str(i+1),fill='black');thumbs.append(canvas)
cols=4;contact=Image.new('RGB',(cols*275,((len(thumbs)+cols-1)//cols)*365),'white')
for i,im in enumerate(thumbs):contact.paste(im,((i%cols)*275,(i//cols)*365))
contact.save(r/'contact.png')
print('Rendered via installed Microsoft Word because packaged renderer dependency is unavailable')
