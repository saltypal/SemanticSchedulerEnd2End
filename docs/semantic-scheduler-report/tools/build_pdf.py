"""Render the Pandoc AST to a bookmarked, paginated PDF using local fonts."""
from pathlib import Path
import json
import math
import re
import textwrap
from xml.sax.saxutils import escape
from PIL import Image as PILImage

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
    Spacer, PageBreak, LongTable, TableStyle, Image, Preformatted, KeepTogether)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path('/workspace/reports/semantic-scheduler')
font_dir = Path('/usr/share/fonts/truetype/dejavu')
for name, filename in [('Body', 'DejaVuSerif.ttf'), ('BodyBold', 'DejaVuSerif-Bold.ttf'),
    ('BodyItalic', 'DejaVuSerif-Italic.ttf'), ('Sans', 'DejaVuSans.ttf'),
    ('SansBold', 'DejaVuSans-Bold.ttf'), ('Mono', 'DejaVuSansMono.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='BodyItalic',boldItalic='BodyBold')
pdfmetrics.registerFontFamily('Sans',normal='Sans',bold='SansBold',italic='Sans',boldItalic='SansBold')
ink = colors.HexColor('#17283c')
accent = colors.HexColor('#246e65')
muted = colors.HexColor('#53647a')
width, height = A4
left = right = 18 * mm
usable = width-left-right
styles = {
 'body': ParagraphStyle('BodyTextLocal',fontName='Body',fontSize=10,leading=15.2,
          textColor=ink,spaceAfter=9,allowWidows=0,allowOrphans=0,splitLongWords=1),
 'h1': ParagraphStyle('Chapter',fontName='SansBold',fontSize=20,leading=26,
          textColor=ink,spaceBefore=4,spaceAfter=16,keepWithNext=True),
 'h2': ParagraphStyle('Subchapter',fontName='SansBold',fontSize=13,leading=18,
          textColor=ink,spaceBefore=15,spaceAfter=8,keepWithNext=True),
 'cell': ParagraphStyle('Cell',fontName='Sans',fontSize=8,leading=11.5,
          textColor=ink,spaceAfter=0,splitLongWords=1),
 'head': ParagraphStyle('HeadCell',fontName='SansBold',fontSize=8,leading=11.5,
          textColor=ink,spaceAfter=0,splitLongWords=1),
 'caption': ParagraphStyle('Caption',fontName='Sans',fontSize=8.4,leading=12,
          textColor=muted,spaceBefore=7,spaceAfter=12),
 'code': ParagraphStyle('CodeLocal',fontName='Mono',fontSize=8,leading=11,
          textColor=ink,backColor=colors.HexColor('#eaf0f4'),borderPadding=8,spaceBefore=5,spaceAfter=12),
}

def inline(items):
    out=[]
    for x in items:
        t=x['t'];c=x.get('c')
        if t=='Str':out.append(escape(c))
        elif t in ('Space','SoftBreak','LineBreak'):out.append(' ')
        elif t=='Code':out.append('<font face="Mono">'+escape(c[1])+'</font>')
        elif t=='Strong':out.append('<b>'+inline(c)+'</b>')
        elif t=='Emph':out.append('<i>'+inline(c)+'</i>')
        elif t=='Link':
            label=inline(c[1]);target=c[2][0]
            out.append('<link href="'+escape(target, {'"':'&quot;'})+'" color="#246e65">'+label+'</link>')
        elif t=='Quoted':out.append('“'+inline(c[1])+'”')
        elif t=='Math':out.append(escape(c[1]))
        elif t=='Span':out.append(inline(c[1]))
        elif t=='Image':out.append(inline(c[1]))
        else:raise ValueError(f'Unsupported inline {t}')
    return ''.join(out)

def plain(items):
    return re.sub('<[^>]+>','',inline(items))

def blocks_text(blocks):
    return '<br/>'.join(inline(b['c']) for b in blocks if b['t'] in ('Plain','Para'))

def para(s,style='body'):
    return Paragraph(s,styles[style])

def figure(node):
    attr,alt,target=node['c']
    source=ROOT/target[0]
    path=source.with_suffix('.png')
    with PILImage.open(path) as source_image:
        image_width,image_height=source_image.size
    image=Image(str(path),width=usable,height=usable*image_height/image_width)
    return KeepTogether([image,para(inline(alt),'caption')])

def table(block):
    attr,caption,colspec,head,bodies,foot=block['c']
    rows=list(head[1])
    headcount=len(rows)
    for body in bodies:rows.extend(body[2]);rows.extend(body[3])
    rows.extend(foot[1])
    raw=[[blocks_text(cell[4]) for cell in row[1]] for row in rows]
    n=len(colspec)
    weights=[]
    for col in range(n):
        lengths=[len(re.sub('<[^>]+>','',r[col])) for r in raw if len(r)>col]
        weights.append(max(2.5,math.sqrt(min(150,max(lengths,default=10)))))
    widths=[usable*w/sum(weights) for w in weights]
    data=[[para(x,'head' if i<headcount else 'cell') for x in row] for i,row in enumerate(raw)]
    grid=LongTable(data,colWidths=widths,repeatRows=headcount,hAlign='LEFT')
    commands=[('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),
      ('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),
      ('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,-1),.35,colors.HexColor('#dbe3eb'))]
    if headcount:commands.append(('BACKGROUND',(0,0),(-1,headcount-1),colors.HexColor('#e7efee')))
    for i in range(headcount,len(data)):
        if (i-headcount)%2==1:commands.append(('BACKGROUND',(0,i),(-1,i),colors.HexColor('#f0f4f7')))
    grid.setStyle(TableStyle(commands))
    return [Spacer(1,6),grid,Spacer(1,12)]

def render(blocks,first_chapter=False):
    result=[]
    for block in blocks:
        t=block['t'];c=block.get('c')
        if t=='Header':
            level,attr,title=c
            if level==1:result.append(PageBreak())
            p=para(inline(title),'h1' if level==1 else 'h2')
            p._level=level;p._anchor=attr[0];p._plain=plain(title)
            result.append(p)
        elif t in ('Para','Plain'):
            if len(c)==1 and c[0]['t']=='Image':result.append(figure(c[0]))
            else:result.append(para(inline(c)))
        elif t=='Table':result.extend(table(block))
        elif t=='Figure':result.extend(render(c[2]))
        elif t=='CodeBlock':
            lines=[]
            for line in c[1].splitlines():
                lines.extend(textwrap.wrap(line,width=89,replace_whitespace=False,
                    drop_whitespace=False,break_long_words=True,break_on_hyphens=False) or [''])
            result.append(Preformatted('\n'.join(lines),styles['code']))
        elif t=='BulletList':
            for item in c:
                if item and item[0]['t'] in ('Plain','Para'):
                    result.append(para('• '+inline(item[0]['c'])))
                    result.extend(render(item[1:]))
                else:result.extend(render(item))
        elif t=='OrderedList':
            attrs,items=c
            for i,item in enumerate(items,attrs[0]):
                if item and item[0]['t'] in ('Plain','Para'):
                    result.append(para(str(i)+'. '+inline(item[0]['c'])))
                    result.extend(render(item[1:]))
                else:result.extend(render(item))
        elif t=='BlockQuote':result.extend(render(c))
        elif t in ('Null','HorizontalRule'):result.append(Spacer(1,10))
        else:raise ValueError(f'Unsupported block {t}')
    return result

def page(canvas,doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#dbe3eb'))
    canvas.line(left,height-16*mm,width-right,height-16*mm)
    canvas.setFont('Sans',8)
    canvas.setFillColor(muted)
    canvas.drawString(left,height-12.5*mm,'THE PICTURE MESSENGER  /  SEMANTIC SCHEDULER')
    canvas.drawString(left,10.5*mm,'Implementation · evidence · proposed research')
    canvas.drawRightString(width-right,10.5*mm,str(doc.page))
    canvas.restoreState()

class Document(BaseDocTemplate):
    def afterFlowable(self,flowable):
        if hasattr(flowable,'_anchor'):
            self.canv.bookmarkPage(flowable._anchor)
            if flowable._level==1:
                self.canv.addOutlineEntry(flowable._plain,flowable._anchor,level=0,closed=False)
                self.notify('TOCEntry',(0,flowable._plain,self.page,flowable._anchor))

doc=Document(str(ROOT/'complete-report.pdf'),pagesize=A4,leftMargin=left,rightMargin=right,
    topMargin=23*mm,bottomMargin=20*mm,title='The Picture Messenger — Complete Scheduler Report',
    author='Project technical report',pageCompression=1)
doc.addPageTemplates(PageTemplate(id='report',frames=[Frame(left,20*mm,usable,height-43*mm,
    leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=page))
cover_title=ParagraphStyle('CoverTitle',fontName='SansBold',fontSize=34,leading=41,textColor=ink,spaceAfter=22)
cover_sub=ParagraphStyle('CoverSubtitle',fontName='Sans',fontSize=16,leading=23,textColor=muted,spaceAfter=24)
story=[Spacer(1,25*mm),Paragraph('The Picture<br/>Messenger',cover_title),
  Paragraph('A complete, plain-language guide to our semantic and urgency driven scheduler',cover_sub),
  para('Simple explanations first. Exact rules, examples, evidence and limitations alongside them.'),
  Spacer(1,10*mm),para('<b>31 chapters · four implemented scheduling policies · static and dynamic designs</b>'),
  para('Includes the six-job replay, seven semantic measurements, both codebooks, image codecs, '
       'additional algorithms, benchmark design, related work and production requirements.'),
  Spacer(1,10*mm),para('<b>Evidence labels</b><br/>EXISTS — implementation inspected<br/>'
       'CHECKED NOW — the stated check was executed<br/>RECORDED EARLIER — historical evidence<br/>'
       'PROPOSED — a specification or research direction'),
  Spacer(1,8*mm),para('Reviewed source commit:<br/><font face="Mono" size="8">'
       'edad7d07c44fd3006e4359b8c32b6dbe143818d7</font>'),PageBreak(),
  para('Contents','h1')]
toc=TableOfContents()
toc.levelStyles=[ParagraphStyle('TOCLevel',fontName='Sans',fontSize=9.2,leading=14,
    leftIndent=0,firstLineIndent=0,spaceBefore=5,textColor=ink)]
story.append(toc)
ast=json.loads((ROOT/'execution/report-ast.json').read_text())
story.extend(render(ast['blocks']))
doc.multiBuild(story)
print('Created',ROOT/'complete-report.pdf')
