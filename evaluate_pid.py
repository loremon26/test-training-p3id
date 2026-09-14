"""Evaluate independent PID2Graph benchmarks, on provided patches and/or repatched full plans."""
import argparse
import json
from pathlib import Path
from metric_pid import PIDMetrics
from predict_image import load_predictor,predict_patches,predict_plan
from pid_graph import parse_graph,overlay


def paired_image(graph):
    paths=[graph.with_suffix(ext) for ext in ('.png','.jpg','.jpeg') if graph.with_suffix(ext).is_file()]
    if len(paths)!=1:
        raise ValueError(f'Expected one paired image: {graph}')
    return paths[0]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True,help='Directory containing Patched and Complete')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--config',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--sources',nargs='+',default=['PID2Graph OPEN100','PID2Graph Synthetic','Dataset PID'])
    p.add_argument('--mode',choices=['patched','stitched','both'],default='both')
    p.add_argument('--device',default='cuda')
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--max-samples',type=int,help='Smoke evaluation only; limit per source and mode')
    p.add_argument('--patch-size',type=int,default=1500)
    p.add_argument('--stride',type=int,default=750)
    p.add_argument('--node-threshold',type=float,default=.05)
    p.add_argument('--edge-threshold',type=float,default=.05)
    p.add_argument('--merge-threshold',type=float,default=.15)
    p.add_argument('--include-borders',action='store_true')
    p.add_argument('--edge-class-mismatch-fn',action='store_true',help='Also count wrong-type predictions as FN; default is literal A1')
    a=p.parse_args()
    if a.batch_size<1 or (a.max_samples is not None and a.max_samples<1):
        p.error('Batch size and sample limit must be positive')
    model,config,device=load_predictor(a.checkpoint,a.config,a.device)
    result={'protocol':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'results':{}}
    for source in a.sources:
        for mode in (['patched','stitched'] if a.mode=='both' else [a.mode]):
            folder=a.dataset_root/('Patched' if mode=='patched' else 'Complete')/source
            graphs=sorted(folder.rglob('*.graphml'))
            if not graphs:
                raise FileNotFoundError(f'No annotations under {folder}')
            if a.max_samples:
                graphs=graphs[:a.max_samples]
            metrics=PIDMetrics(a.include_borders,a.edge_class_mismatch_fn)
            for start in range(0,len(graphs),a.batch_size if mode=='patched' else 1):
                group=graphs[start:start+(a.batch_size if mode=='patched' else 1)]
                if mode=='patched':
                    predictions=predict_patches(model,config,[paired_image(g) for g in group],device,
                                                a.node_threshold,a.edge_threshold)
                else:
                    g=group[0]
                    predictions=[predict_plan(model,config,paired_image(g),a.output_dir/source/'stitched'/g.stem,
                        device,patch_size=a.patch_size,stride=a.stride,batch_size=a.batch_size,
                        node_threshold=a.node_threshold,edge_threshold=a.edge_threshold,merge_threshold=a.merge_threshold)]
                for g,prediction in zip(group,predictions):
                    ns,es=parse_graph(g)
                    metrics.add(dict(nodes=ns,edges=es),prediction)
                if start==0 and mode=='patched':
                    overlay(paired_image(group[0]),predictions[0]['nodes'],predictions[0]['edges'],
                            a.output_dir/source/'patched_example.png')
                if start%100==0:
                    print(f'{source}/{mode}: {start+len(group)}/{len(graphs)}',flush=True)
            result['results'][f'{source}/{mode}']=metrics.compute()
            a.output_dir.mkdir(parents=True,exist_ok=True)
            (a.output_dir/'metrics.json').write_text(json.dumps(result,indent=2))
            print(json.dumps(result['results'][f'{source}/{mode}']),flush=True)


if __name__=='__main__':
    main()
