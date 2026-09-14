"""Approximate III-D synthetic generator: public templates, connected orthogonal graphs."""
import argparse
from collections import Counter
import copy
import hashlib
import io
import json
import random
import re
import sys
from pathlib import Path
from urllib.request import urlopen
from xml.etree import ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from pid_graph import center, box, point_node, draw_line, write_graph, clip_segment

SOURCE='https://gitlab.com/baltakatei/BK-2020-04'
API='https://gitlab.com/api/v4/projects/21329174'


def prepare_templates(directory):
    """Extract labelled symbol groups with inherited transforms; retain source attribution."""
    import cairosvg
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'templates.json').exists():
        return json.loads((directory/'templates.json').read_text())
    with urlopen(API+'/repository/commits/master',timeout=60) as response:
        revision=json.load(response)['id']
    with urlopen(API+'/repository/files/README.org/raw?ref='+revision,timeout=60) as response:
        license_text=response.read().decode()
    (directory/'SOURCE_README.org').write_text(license_text)
    records=[]
    for sheet in range(1,8):
        path=f'DWG%2FBK-2020-04-PID-1-SHT{sheet}.svg'
        with urlopen(API+f'/repository/files/{path}/raw?ref={revision}',timeout=60) as response:
            raw=response.read()
        (directory/f'sheet{sheet}.svg').write_bytes(raw)
        root=ET.fromstring(raw)
        parents={child:parent for parent in root.iter() for child in parent}
        # Some editions use http://www.inkscape.org/namespaces/inkscape.
        def label(element):
            return next((v for k,v in element.attrib.items() if k.endswith('}label')), '')
        for symbol in root.iter():
            match=re.fullmatch(r'REG#:(.+)_symbol',label(symbol))
            if not match:
                continue
            chain=[]
            parent=parents.get(symbol)
            group=None
            while parent is not None and parent is not root:
                chain.append(parent)
                m=re.fullmatch(r'Main_group_(\d+)',label(parent))
                if m:
                    group=int(m[1])
                parent=parents.get(parent)
            # Pipe fragments/internal parts are not complete equipment templates.
            if group is None or group>=24:
                continue
            cls='tank' if group==1 else 'pump' if group in (15,16,17) else 'valve' if group in (21,22,23) else 'general'
            isolated=ET.Element(root.tag,root.attrib)
            for child in root:
                if child.tag.endswith('}defs'):
                    isolated.append(copy.deepcopy(child))
            holder=isolated
            for ancestor in reversed(chain):
                holder=ET.SubElement(holder,ancestor.tag,ancestor.attrib)
            holder.append(copy.deepcopy(symbol))
            png=cairosvg.svg2png(bytestring=ET.tostring(isolated),output_width=3364,background_color='white')
            im=Image.open(io.BytesIO(png)).convert('L')
            bounds=ImageOps.invert(im).getbbox()
            if bounds is None:
                raise ValueError(f'Empty rendered template: sheet {sheet}, {match[1]}')
            name=f'{cls}_{sheet}_{match[1]}.png'
            im.crop(bounds).save(directory/name)
            records.append(dict(file=name,label=cls,source=SOURCE,revision=revision,
                sheet=sheet,registration=match[1],sha256=hashlib.sha256(raw).hexdigest(),
                license='CC-BY-SA-4.0',author='Steven Baltakatei Sandoval'))
        print(f'Sheet {sheet}: {len(records)} public templates extracted',flush=True)
    # Explicitly identified original procedural additions for absent classes.
    for cls in ('instrumentation','inlet_outlet','arrow'):
        for variant in range(8):
            im=Image.new('L',(100,100),255)
            draw=ImageDraw.Draw(im)
            if cls=='instrumentation':
                draw.ellipse((12,12,88,88),outline=0,width=2)
                if variant%3:
                    draw.line((12,50,88,50),fill=0,width=2)
                draw.text((39,30),['PI','TI','FI','LI','PC','TC','FC','LC'][variant],fill=0)
            elif cls=='inlet_outlet':
                draw.polygon([(10,30),(65,30),(90,50),(65,70),(10,70)],outline=0,width=2)
                draw.text((20,43),str(variant+1),fill=0)
            else:
                draw.line((10,50,90,50),fill=0,width=2)
                draw.line((65,30,90,50,65,70),fill=0,width=2)
                im=im.rotate(90*(variant%4))
            im=im.crop(ImageOps.invert(im).getbbox())
            name=f'{cls}_procedural_{variant}.png'
            im.save(directory/name)
            records.append(dict(file=name,label=cls,source='procedural in this repository',license='CC0-1.0'))
    if set(r['label'] for r in records) != {'general','tank','valve','pump','instrumentation','inlet_outlet','arrow'}:
        raise ValueError('Template extraction did not cover all seven classes')
    manifest=dict(source=SOURCE,revision=revision,license='CC-BY-SA-4.0 for public templates and derived drawings',
                  templates=records,approximation='Public ISO equipment + procedural instrumentation, ports and arrows; not Synthetic 700')
    (directory/'templates.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def generate(seed, template_dir, templates, width=7000, height=4500, symbol_count=42):
    if width<512 or height<512 or symbol_count<7 or symbol_count>150:
        raise ValueError('Require dimensions >=512 and 7..150 symbols per drawing')
    rng=random.Random(seed)
    by_class={c:[t for t in templates if t['label']==c] for c in sorted({t['label'] for t in templates})}
    classes=list(by_class)
    weights=[{'general':.48,'valve':.19,'tank':.10,'instrumentation':.12,'pump':.06,'inlet_outlet':.03,'arrow':.02}[c] for c in classes]
    cols=int(np.ceil(np.sqrt(symbol_count*width/height)))
    rows=int(np.ceil(symbol_count/cols))
    slots=[(c,r) for r in range(rows) for c in range(cols)]
    rng.shuffle(slots)
    nodes, sprites = [], []
    labels=classes+rng.choices(classes,weights,k=symbol_count-len(classes))
    rng.shuffle(labels)
    for i,((c,r),cls) in enumerate(zip(slots,labels)):
        x=(c+.5+rng.uniform(-.18,.18))*width/cols
        y=(r+.5+rng.uniform(-.18,.18))*height/rows
        template=rng.choice(by_class[cls])
        with Image.open(Path(template_dir)/template['file']) as src:
            sprite=src.convert('L')
        maximum=max(14,min(240,min(width/cols,height/rows)*.30))
        size=max(12,min(maximum,rng.lognormvariate(4.4,.5)))
        factor=size/max(sprite.size)
        sprite=sprite.resize((max(4,round(sprite.width*factor)),max(4,round(sprite.height*factor))),Image.Resampling.LANCZOS)
        if rng.random()<.35:
            sprite=sprite.transpose(Image.Transpose.ROTATE_90)
        x,y=round(x),round(y)
        x1,y1=x-sprite.width//2,y-sprite.height//2
        nodes.append(dict(id=str(i),label=cls,xmin=float(x1),ymin=float(y1),
                          xmax=float(x1+sprite.width),ymax=float(y1+sprite.height),score=1.))
        sprites.append((sprite,(x1,y1)))
    points=[center(n) for n in nodes]
    # Euclidean minimum spanning tree, plus two local cycles.
    pairs=sorted((float(np.linalg.norm(points[i]-points[j])),i,j) for i in range(len(nodes)) for j in range(i))
    components=list(range(len(nodes)))
    links=[]
    for distance,i,j in pairs:
        if components[i]!=components[j]:
            old,new=components[j],components[i]
            components=[new if k==old else k for k in components]
            links.append((i,j))
    candidates=[(i,j) for _,i,j in pairs[:symbol_count*2] if (i,j) not in links]
    links+=rng.sample(candidates,min(2,len(candidates)))
    segments=[]
    for i,j in links:
        a,b=points[i],points[j]
        options=[(a,np.array([(a[0]+b[0])/2,a[1]]),np.array([(a[0]+b[0])/2,b[1]]),b),
                 (a,np.array([a[0],(a[1]+b[1])/2]),np.array([b[0],(a[1]+b[1])/2]),b)]
        def obstacles(route):
            return sum(clip_segment(u,v,box(n)) is not None for u,v in zip(route,route[1:])
                       for k,n in enumerate(nodes) if k not in (i,j))
        route=min(options,key=obstacles)
        if obstacles(route):
            # A new seed/layout is safer than drawing lines through unrelated symbols.
            raise ValueError('route intersects equipment')
        label='non_solid' if rng.random()<.16 else 'solid'
        segments.extend((np.asarray(u),np.asarray(v),label) for u,v in zip(route,route[1:]) if np.linalg.norm(v-u)>1e-6)
    coords={tuple(np.round(center(n),6)):n['id'] for n in nodes}
    def node_at(xy):
        key=tuple(np.round(xy,6))
        if key not in coords:
            coords[key]=str(len(nodes))
            nodes.append(point_node(coords[key],xy,'ankle'))
        return coords[key]
    split_points=[[a,b] for a,b,_ in segments]
    for i,(a,b,_) in enumerate(segments):
        for j,(c,d,_) in enumerate(segments[:i]):
            horizontal=abs(a[1]-b[1])<1e-9
            other_horizontal=abs(c[1]-d[1])<1e-9
            if horizontal!=other_horizontal:
                xy=np.array([c[0],a[1]]) if horizontal else np.array([a[0],c[1]])
                if (np.minimum(a,b)-1e-6<=xy).all() and (xy<=np.maximum(a,b)+1e-6).all() and (np.minimum(c,d)-1e-6<=xy).all() and (xy<=np.maximum(c,d)+1e-6).all():
                    split_points[i].append(xy)
                    split_points[j].append(xy)
            elif abs(a[1 if horizontal else 0]-c[1 if horizontal else 0])<1e-6:
                for xy in (a,b,c,d):
                    if (np.minimum(a,b)-1e-6<=xy).all() and (xy<=np.maximum(a,b)+1e-6).all():
                        split_points[i].append(xy)
                    if (np.minimum(c,d)-1e-6<=xy).all() and (xy<=np.maximum(c,d)+1e-6).all():
                        split_points[j].append(xy)
    edges={}
    for (a,b,label),cuts in zip(segments,split_points):
        cuts=sorted({tuple(p) for p in cuts},key=lambda p: float((np.array(p)-a)@(b-a)))
        for u,v in zip(cuts,cuts[1:]):
            ends=tuple(sorted((node_at(u),node_at(v))))
            if ends[0]!=ends[1]:
                chosen='solid' if ends in edges and edges[ends]['label']=='solid' else label
                edges[ends]=dict(source=ends[0],target=ends[1],label=chosen,score=1.)
    degree=Counter(k for e in edges.values() for k in (e['source'],e['target']))
    for n in nodes[symbol_count:]:
        if degree[n['id']]>=3:
            n['label']='crossing'
    image=Image.new('L',(width,height),255)
    draw=ImageDraw.Draw(image)
    lookup={n['id']:n for n in nodes}
    adjacent={k:set() for k in lookup}
    for e in edges.values():
        adjacent[e['source']].add(e['target'])
        adjacent[e['target']].add(e['source'])
        draw_line(draw,center(lookup[e['source']]),center(lookup[e['target']]),e['label'],0,width=rng.choice([2,3]))
    visited=set()
    pending=[nodes[0]['id']]
    while pending:
        k=pending.pop()
        if k not in visited:
            visited.add(k)
            pending.extend(adjacent[k]-visited)
    if len(visited)!=len(nodes):
        raise RuntimeError('Generator produced a disconnected graph')
    for sprite,xy in sprites:
        image.paste(sprite,xy)
    return image,nodes,list(edges.values())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=Path('data/generated/Complete'))
    p.add_argument('--templates',type=Path,default=Path('data/templates'))
    p.add_argument('--prepare-templates',action='store_true',help='Download licensed public SVGs and extract PNG symbols')
    p.add_argument('--count',type=int,default=2000)
    p.add_argument('--seed',type=int,default=10)
    p.add_argument('--width',type=int,default=7000)
    p.add_argument('--height',type=int,default=4500)
    p.add_argument('--symbols',type=int,default=42)
    a=p.parse_args()
    if a.count<0:
        p.error('--count cannot be negative')
    library=prepare_templates(a.templates) if a.prepare_templates else json.loads((a.templates/'templates.json').read_text())
    a.output_dir.mkdir(parents=True,exist_ok=True)
    if (a.output_dir/'manifest.json').exists():
        raise FileExistsError('Use a fresh output directory to preserve previous drawings')
    records=[]
    for index in range(a.count):
        stem=f'synthetic_{index:05d}'
        for attempt in range(100):
            seed=a.seed+index*1000+attempt
            try:
                image,nodes,edges=generate(seed,a.templates,library['templates'],a.width,a.height,a.symbols)
                break
            except ValueError as error:
                if str(error)!='route intersects equipment':
                    raise
        else:
            raise RuntimeError(f'Could not route drawing {index}; lower symbol count')
        if (a.output_dir/f'{stem}.png').exists():
            raise FileExistsError(stem)
        image.save(a.output_dir/f'{stem}.png')
        write_graph(a.output_dir/f'{stem}.graphml',nodes,edges)
        records.append(dict(id=stem,seed=seed,image=f'{stem}.png',graph=f'{stem}.graphml',
                            nodes=len(nodes),edges=len(edges),node_classes=dict(Counter(n['label'] for n in nodes)),
                            edge_classes=dict(Counter(e['label'] for e in edges))))
        if (index+1)%20==0 or index+1==a.count:
            print(f'{index+1}/{a.count} drawings',flush=True)
    manifest=dict(schema_version=1,seed=a.seed,parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                  template_library=library,drawings=records)
    (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
