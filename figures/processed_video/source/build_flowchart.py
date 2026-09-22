"""Vector PDF and editable draw.io for the actual EvDetMAV pipeline.
No synthetic/video frames are included in this algorithm-only diagram.
"""
from pathlib import Path
import re, html, xml.etree.ElementTree as ET
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'figures/processed_video'
pdfmetrics.registerFont(TTFont('Songti',str(ROOT/'figures/fonts/SongtiSC-Regular.ttf')))
pdfmetrics.registerFont(TTFont('TNR','/System/Library/Fonts/Supplemental/Times New Roman.ttf'))
W,H=1080,780
c=canvas.Canvas(str(ROOT/'output/pdf/fig4_algorithm_flow.pdf'),pagesize=(W*.7,H*.7))
c.setTitle('图4 事件流旋翼检测流程');c.setAuthor('EvDetMAV implementation diagram')
c.scale(.7,.7)
mx=ET.Element('mxfile',host='app.diagrams.net',version='26.0.0')
diagram=ET.SubElement(mx,'diagram',id='evdetmav-flow',name='图4 算法流程')
model=ET.SubElement(diagram,'mxGraphModel',dx='1080',dy='780',grid='1',gridSize='10',page='1',pageScale='1',pageWidth=str(W),pageHeight=str(H),math='0',shadow='0')
rt=ET.SubElement(model,'root');ET.SubElement(rt,'mxCell',id='0');ET.SubElement(rt,'mxCell',id='1',parent='0')
nodes={}; counter=0
INK='#203040'; GRAY='#657382'; LINE='#718294'
def font(ch): return 'TNR' if ord(ch)<0x2e80 else 'Songti'
def runs(s):
 out=[]
 for ch in s:
  f=font(ch)
  if out and out[-1][0]==f:out[-1]=(f,out[-1][1]+ch)
  else:out.append((f,ch))
 return out
def text(x,y,s,size=16,color=INK,center=True):
 for j,line in enumerate(s.split('\n')):
  rr=runs(line);width=sum(pdfmetrics.stringWidth(t,f,size) for f,t in rr)
  xx=x-width/2 if center else x
  for f,t in rr:
   c.setFont(f,size);c.setFillColor(HexColor(color));c.drawString(xx,H-y-j*size*1.32,t);xx+=pdfmetrics.stringWidth(t,f,size)
def val(s):
 return '<div style="font-family:Times New Roman">'+ '<br>'.join(''.join('<span style="font-family:'+('Times New Roman' if f=='TNR' else 'Songti SC')+'">'+html.escape(t)+'</span>' for f,t in runs(line)) for line in s.split('\n'))+'</div>'
def cell(id,x,y,w,h,s,style):
 ce=ET.SubElement(rt,'mxCell',id=id,value=val(s),style='html=1;whiteSpace=wrap;fontFamily=Times New Roman;fontSize=16;fontColor='+INK+';'+style,vertex='1',parent='1')
 ET.SubElement(ce,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
def label(id,x,y,w,h,s,size=16,color=INK):
 cell(id,x,y,w,h,s,f'text;strokeColor=none;fillColor=none;fontSize={size};fontColor={color}')
 text(x+w/2,y+h/2+size*.32-(len(s.split('\n'))-1)*size*.66,s,size,color)
def box(id,x,y,w,h,s,fill='#FFFFFF',stroke='#B6C5D2',size=16):
 nodes[id]=(x,y,w,h)
 c.setFillColor(HexColor(fill));c.setStrokeColor(HexColor(stroke));c.setLineWidth(1.1);c.roundRect(x,H-y-h,w,h,7,stroke=1,fill=1)
 text(x+w/2,y+h/2+size*.32-(len(s.split('\n'))-1)*size*.66,s,size)
 cell(id,x,y,w,h,s,f'rounded=1;arcSize=10;fillColor={fill};strokeColor={stroke};fontSize={size}')
def arrow(id,source,target,points=None,labeltxt=None):
 sx,sy,sw,sh=nodes[source];tx,ty,tw,th=nodes[target]
 if points is None:
  if abs(sx-tx)<30: points=[(sx+sw/2,sy+sh),(tx+tw/2,ty)]
  else:points=[(sx+sw,sy+sh/2),(tx,ty+th/2)]
 c.setStrokeColor(HexColor(LINE));c.setFillColor(HexColor(LINE));c.setLineWidth(1.4)
 p=c.beginPath();p.moveTo(points[0][0],H-points[0][1])
 for x,y in points[1:]:p.lineTo(x,H-y)
 c.drawPath(p)
 import math
 x,y=points[-1];px,py=points[-2];a=math.atan2(y-py,x-px);r=7
 p=c.beginPath();p.moveTo(x,H-y);p.lineTo(x-r*math.cos(a-.45),H-(y-r*math.sin(a-.45)));p.lineTo(x-r*math.cos(a+.45),H-(y-r*math.sin(a+.45)));p.close();c.drawPath(p,fill=1,stroke=0)
 ce=ET.SubElement(rt,'mxCell',id=id,value=val(labeltxt or ''),style=f'edgeStyle=orthogonalEdgeStyle;html=1;endArrow=block;strokeColor={LINE};fontFamily=Times New Roman;fontSize=13;',edge='1',parent='1',source=source,target=target)
 geo=ET.SubElement(ce,'mxGeometry',relative='1',attrib={'as':'geometry'})
 if len(points)>2:
  arr=ET.SubElement(geo,'Array',attrib={'as':'points'})
  for x,y in points[1:-1]:ET.SubElement(arr,'mxPoint',x=str(x),y=str(y))
 if labeltxt:
  xx=(points[0][0]+points[-1][0])/2;yy=(points[0][1]+points[-1][1])/2-8;text(xx,yy,labeltxt,13)
# Header
label('title',30,18,1020,35,'事件流旋翼检测：单窗计算流程',24)
label('subtitle',30,57,1020,23,'显著性定位 → 局部峰谷筛选 → 由粗到细区域输出',16,GRAY)
box('input',30,103,245,55,'原始事件 e = (x, y, t, p)\n文件读取与时间、极性统一',size=16)
box('window',375,103,300,55,'时间窗 ΔT 内的全部原始事件\n默认每个文件作为一个检测窗',size=16)
box('scope',775,103,275,55,'各窗独立计算\n不包含跨窗跟踪与持续性评分',fill='#F4F6F8',size=15)
arrow('e-input','input','window')
# Stage containers
for id,x,title,fill,stroke in [('s1',30,'01  密度感知显著性','#F3F7FC','#7A9CBD'),('s2',375,'02  局部时序评分','#F1F7F5','#75A89A'),('s3',720,'03  高斯一致性细化','#FBF6EF','#C0A174')]:
 c.setFillColor(HexColor(fill));c.setStrokeColor(HexColor(stroke));c.roundRect(x,H-205-358,330,358,10,fill=1,stroke=1)
 cell(id,x,205,330,358,'',f'rounded=1;fillColor={fill};strokeColor={stroke}')
 label(id+'title',x+10,219,310,25,title,19)
box('slices',50,260,290,40,'全视场时间切片：n = 10')
box('binary',50,314,290,49,'正、负二值事件图\n分别膨胀 r = 1 后求交集')
box('sum',50,377,290,49,'逐片交集累加并缩放至 0–255\n像素显著阈值 τs = 50')
box('candidates',50,440,290,49,'8 邻域连通域 → 矩形距离合并\n距离 ≤ 8 px，候选框扩边 4 px')
box('topk',50,503,290,40,'显著值总和排序，取前 K = 4')
arrow('e-window','window','slices',[(525,158),(525,183),(195,183),(195,260)])
for i,(a,b) in enumerate(zip(['slices','binary','sum','candidates'],['binary','sum','candidates','topk'])):arrow(f'e-sal-{i}',a,b)
box('roi',395,260,290,40,'按候选框提取原始局部事件')
box('mslices',395,314,290,40,'重新切成 m = 12 个时间片')
box('features',395,368,290,63,'正事件数量 fd\n相邻片结构相似度 fs\n相邻片主方向相似度 fp',size=15)
box('smooth',395,445,290,40,'有效值 → 均值平滑 → 归一化')
box('score',395,499,290,44,'逐特征：有峰 +1，有谷 +1\n总分 sp：0–6',size=15)
arrow('e-roi','topk','roi',[(340,523),(357,523),(357,280),(395,280)])
for i,(a,b) in enumerate(zip(['roi','mslices','features','smooth'],['mslices','features','smooth','score'])):arrow(f'e-period-{i}',a,b)
box('fine',740,260,290,49,'sp ≥ 3 的候选进入细化\n显著阈值 τf = 50，再取连通域')
box('gauss',740,326,290,49,'显著值加权拟合二维高斯\n观测值与预测值的余弦相似度 g')
box('keep',740,392,290,49,'保留 g ≥ 0.25 的细区域\n合并其掩码，求外接矩形')
box('fallback',740,458,290,69,'若没有细区域通过：\n默认回退至非空粗掩码及原候选框\n关闭回退则删除该候选',fill='#FFFDF8',size=15)
arrow('e-pass','score','fine',[(685,521),(704,521),(704,284),(740,284)])
for i,(a,b) in enumerate(zip(['fine','gauss','keep'],['gauss','keep','fallback'])):arrow(f'e-fine-{i}',a,b)
box('reject',395,583,290,38,'sp < 3：丢弃候选',fill='#F5F5F5',size=15)
arrow('e-reject','score','reject')
box('output',740,584,290,60,'默认：逐候选 propeller_* 框\n可选：全部合为一个 mav 框\n掩码 + 分数 + CSV + 可视化',fill='#EDF3F8',size=15)
arrow('e-output','fallback','output')
label('note',30,660,1020,45,'注：高斯回退允许低于阈值的粗区域输出；峰谷评分不等同于严格周期检验。\n视频素材用于过程展示；精确时间切片和评分需对应原始事件及处理结果。',15,GRAY)
label('footer',30,725,1020,25,'图 4  当前 evdetmav 实现的算法流程（默认参数）',17)
c.showPage();c.save()
ET.indent(mx,space='  ');ET.ElementTree(mx).write(OUT/'fig4_algorithm_flow.drawio',encoding='utf-8',xml_declaration=True)
print(ROOT/'output/pdf/fig4_algorithm_flow.pdf')
print(OUT/'fig4_algorithm_flow.drawio')
