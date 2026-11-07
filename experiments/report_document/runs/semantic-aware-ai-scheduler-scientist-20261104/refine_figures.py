from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle
F=Path(__file__).resolve().parent/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
ink='#23384e';blue='#e7eef8';teal='#def2eb';orange='#fff0d9';gray='#f0f3f6'
def canvas(w,h):
 f,a=plt.subplots(figsize=(w,h));a.set(xlim=(0,w),ylim=(0,h));a.axis('off');return f,a
def box(a,x,y,w,h,t,fc=blue,size=11,lw=1.3):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.015,rounding_size=.07',fc=fc,ec='#587087',lw=lw));a.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=size,color=ink,linespacing=1.45)
def arrow(a,start,end,label=None,style='-',color=ink):
 a.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=16,lw=1.5,linestyle=style,color=color))
 if label:a.text((start[0]+end[0])/2,(start[1]+end[1])/2+.18,label,ha='center',fontsize=9,color=color)
def save(f,name):
 for fmt in ['png','svg']:f.savefig(F/(name+'.'+fmt),dpi=240,bbox_inches='tight',facecolor='white')
 plt.close(f)
f,a=canvas(13,6)
for x,w,title in [(.1,3.8,'SENDER'),(4.2,3.1,'COMPLETE MESSAGE'),(7.65,5.2,'RECEIVER')]:
 a.add_patch(FancyBboxPatch((x,.25),w,5.4,boxstyle='round,pad=.02,rounding_size=.12',fc='white',ec='#c4cfdb',lw=1.2));a.text(x+.16,5.3,title,weight='bold',fontsize=12,color=ink)
box(a,.5,3.8,3,.65,'RGB observation',gray)
box(a,.5,2.35,1.3,.85,'Semantic\nparameters',teal,10)
box(a,2,2.35,1.5,.85,'SQ-GAN\nencoder',blue,10)
arrow(a,(1.2,3.8),(1.2,3.2));arrow(a,(2.75,3.8),(2.75,3.2))
box(a,.5,.8,3,.85,'Serialize parameters and\nquantized latent representation',gray,10)
arrow(a,(1.2,2.35),(1.2,1.65));arrow(a,(2.75,2.35),(2.75,1.65))
box(a,4.5,3.1,2.5,1,'Scheduling header',teal)
box(a,4.5,1.45,2.5,1.4,'Digital SQ01 payload\nquantized codes and\nrequired codec framing',blue,10)
arrow(a,(3.5,1.2),(4.5,1.9));a.text(5.75,.7,'Header + variable-length payload',ha='center',fontsize=9,color=ink)
box(a,8,3.6,4.5,.8,'Buffer of complete received messages',gray)
box(a,8,2.1,2,.85,'One configured\nscheduling policy',teal,10)
box(a,10.5,2.1,2,.85,'Select one\nmessage',orange,10)
arrow(a,(7,3.6),(8,4));arrow(a,(9,3.6),(9,2.95),'headers');arrow(a,(10,2.5),(10.5,2.5))
box(a,8,.65,4.5,.8,'SQ-GAN decoder → reconstructed RGB',blue,11)
arrow(a,(11.5,2.1),(11.5,1.45),'one payload')
save(f,'architecture')

f,a=canvas(13,6.5)
a.text(.1,6.15,'ONE SELECTOR PER EXPERIMENT  ·  ONE MESSAGE PER DISPATCH',fontsize=13,weight='bold',color=ink)
box(a,.15,4.25,2.5,1.2,'Sender processing\nparameters +\nSQ-GAN latent',blue)
arrow(a,(2.65,4.85),(3.7,4.85),'arrives')
a.add_patch(FancyBboxPatch((3.7,1.9),4.9,3.7,boxstyle='round,pad=.04,rounding_size=.12',fc='#f8fafc',ec='#587087',lw=1.5));a.text(6.15,5.25,'RECEIVER BUFFER',ha='center',weight='bold',color=ink)
for y,k,t,color in [(4.2,'A','waiting',gray),(3.25,'B','selected',orange),(2.3,'C','waiting',gray)]:
 box(a,4,y,1.8,.65,'Header '+k,teal,11,2 if k=='B' else 1)
 box(a,5.9,y,2.35,.65,'Payload '+k,color,11,2 if k=='B' else 1)
box(a,3.9,.25,4.45,1.05,'ONE ALGORITHM\nFIFO  OR  SPI-first  OR  LambdaMART  OR …',teal,10)
arrow(a,(4.55,2.3),(4.55,1.3),'read headers')
arrow(a,(7.7,1.3),(7.7,2.3),'selected identity')
box(a,10,3.25,2.7,1.1,'SQ-GAN decoder\nprocesses payload B',blue)
arrow(a,(8.25,3.57),(10,3.8),'dispatch B')
box(a,10,.85,2.7,1.05,'Reconstructed\nimage B',gray)
arrow(a,(11.35,3.25),(11.35,1.9))
a.text(.2,2.9,'A and C remain buffered.\nNew arrivals may join\nwhile B is being decoded.',fontsize=11,color=ink,linespacing=1.6)
a.text(8.8,.25,'Next selection occurs only\nafter decoder completion.',fontsize=10,color=ink)
save(f,'receiver_dispatch')

f,a=canvas(13,7)
a.text(.1,6.65,'64-BYTE SCHEDULING HEADER',weight='bold',fontsize=15,color=ink)
a.text(.1,6.18,'Proposed fixed layout  |  codec and vocabulary profiles shared by both endpoints',fontsize=11,color=ink)
x0=1.3;scale=.69
for i in range(17):a.text(x0+i*scale,5.6,str(i) if i<16 else '16',ha='center',fontsize=9,color='#5a6e81')
layout=[[(0,4,'Protocol / profile',gray),(4,4,'Source ID',blue),(8,4,'Session ID',blue),(12,4,'Message ID',blue)],[(0,8,'Capture timestamp (ms)',gray),(8,4,'Allowed delay (ms)',gray),(12,4,'Payload length',blue)],[(0,4,'Codec / shape profile',blue),(4,4,'SPI · relevance · novelty · status',teal),(8,4,'Scene descriptor 1',orange),(12,4,'Scene descriptor 2',orange)],[(0,4,'Scene descriptor 3',orange),(4,4,'Scene descriptor 4',orange),(8,4,'Scene descriptor 5',orange),(12,4,'Scene descriptor 6',orange)]]
for row,segments in enumerate(layout):
 y=4.5-row*.85;a.text(.12,y+.32,f'{row*16:02d}–{row*16+15:02d}',ha='left',va='center',fontsize=11,color=ink)
 for start,length,label,color in segments:
  x=x0+start*scale;a.add_patch(Rectangle((x,y),length*scale,.64,fc=color,ec='#63798d',lw=1.1));a.text(x+length*scale/2,y+.32,label,ha='center',va='center',fontsize=8.3 if label.startswith('SPI') else 9.5,color=ink,wrap=True)
a.text(.15,5.6,'Byte offsets',fontsize=9,color='#5a6e81')
box(a,1.3,.8,11.04,.7,'Variable-length SQ01 latent payload follows the 64-byte header',blue,12)
a.text(1.3,.3,'Scene descriptor = category + count + area + validity/status. Full 93-class arrays are not inside this reduced profile.',fontsize=9.5,color=ink)
save(f,'header_packet')
print('Three technical diagrams exported as PNG and SVG')
