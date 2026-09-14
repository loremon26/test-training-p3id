"""Functionality for 2D road network dataset."""

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as tvf
from PIL import Image
import numpy as np

import os
import time
import pickle
import random
import yaml
import json
import hashlib
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from xml.etree import ElementTree


from prepare_pid2graph_paper_dataset import (
    PID_NODE_CLASS_TO_ID, PID_NODE_LABEL_ALIASES, PID_EDGE_CLASS_TO_ID,
    PID_EDGE_LABEL_ALIASES, parse_graph,
)

PID_CACHE_SCHEMA_VERSION = "v5"


class ToulouseRoadNetworkDataset(Dataset):
    """
    Generates a subclass of the PyTorch torch.utils.data.Dataset class
    """

    def __init__(self, root_path="data/", split="valid", use_raw_images=False):
        """
        :param root_path: root data path
        :param split: data split in {"train", "valid", "test", "augment"}
        :param max_prev_node: only return the last previous 'max_prev_node' elements in the adjacency row of a node
            default is 4, which corresponds to the 95th percentile in the data
        :param step: step size used in the data generation, default is 0.001° (around 110 metres per datapoint)
        :param use_raw_images: loads raw images if yes, otherwise faster and more compact numpy array representations
        :param return_coordinates: returns coordinates on the real map for each datapoint, used for qualitative studies
        """
        assert split in {"train", "valid", "test", "augment"}
        print(f"Started loading the data ({split})...")
        start_time = time.time()

        dataset_path = f"{root_path}/{split}.pickle"
        images_path = f"{root_path}/{split}_images.pickle"
        images_raw_path = f"{root_path}/{split}/images/"

        ids, list_nodes, list_edges = load_dataset(dataset_path)

        self.ids = ["{:0>7d}".format(int(i)) for i in ids]
        self.nodes = list_nodes
        self.edges = list_edges

        print(f"Started loading the images...")

        if use_raw_images:
            self.images = load_raw_images(ids, images_raw_path)
        else:
            self.images = load_images(ids, images_path)

        print(f"Dataset loading completed, took {round(time.time() - start_time, 2)} seconds!")
        print(f"Dataset size: {len(self)}\n")

    def __len__(self):
        r"""
        :return: data length
        """
        return len(self.ids)

    def __getitem__(self, idx):
        r"""
        :param idx: index in the data
        :return: chosen data point
        """
        return self.images[idx][None], self.nodes[idx], self.edges[idx], self.ids[idx]


class PatchedPIDDataset(Dataset):
    """Loads patched P&ID PNG images and their paired GraphML annotations."""

    SPLITS = {"train", "valid", "test"}

    def __init__(
        self,
        root_path,
        split="train",
        image_size=(512, 512),
        split_seed=10,
        max_nodes=None,
        cache_dir=None,
        train_sources=None,
        test_sources=None,
        validation_percent=5,
        source_limits=None,
        augment=False,
        cache_images=False,
    ):
        if split not in self.SPLITS:
            raise ValueError(f"Unsupported P&ID split: {split}")

        root = Path(root_path)
        if not root.is_dir():
            raise FileNotFoundError(f"P&ID dataset directory does not exist: {root}")

        cache_dir = resolve_cache_dir(cache_dir)
        self.image_size = tuple(image_size)
        samples = index_pid_samples(
            root,
            split_seed,
            max_nodes,
            cache_dir,
            train_sources=train_sources,
            test_sources=test_sources,
            validation_percent=validation_percent,
            source_limits=source_limits,
        )[split]

        if not samples:
            raise RuntimeError(f"No paired P&ID samples found for the {split} split under {root}")

        self.ids = [sample_id for _, _, sample_id in samples]
        self.samples = samples
        self.augment = augment and split == 'train'
        self.epoch = 0
        self.seed = split_seed
        self.metadata = {}
        for _, graph_path, _ in samples:
            manifest_path = graph_path.parent / 'manifest.json'
            if manifest_path.is_file() and str(graph_path.parent) not in self.metadata:
                self.metadata[str(graph_path.parent)] = json.loads(manifest_path.read_text())
        cache_key = hashlib.sha1(
            f"{root.resolve()}:{split_seed}:{max_nodes}:{split}:{self.image_size}:"
            f"{train_sources}:{test_sources}:{validation_percent}:{PID_CACHE_SCHEMA_VERSION}".encode()
        ).hexdigest()[:16]
        self.images, self.graphs = None, None
        if cache_images:
            digest = hashlib.sha1(str([(str(i),i.stat().st_mtime_ns,str(g),g.stat().st_mtime_ns) for i,g,_ in samples]).encode()).hexdigest()[:12]
            self.images, self.graphs = preprocess_pid_samples(
                samples, self.image_size, cache_dir / f"pid_{cache_key}_{digest}_{split}"
            )

        print(f"Loaded {len(self.ids)} P&ID samples for {split}.")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        repeat = 0
        if isinstance(idx, tuple):
            idx, repeat = idx
        if self.images is None:
            image_path, graph_path, _ = self.samples[idx]
            with Image.open(image_path) as source:
                width,height = source.size
                image = source.convert('L').resize(self.image_size, Image.Resampling.BILINEAR)
            graph = load_pid_graph(graph_path,width,height)
        else:
            image = Image.fromarray(np.array(self.images[idx]))
            graph = self.graphs[idx]
        nodes, edges, node_classes, edge_classes, node_boxes = graph
        if self.augment:
            image, nodes, edges, node_classes, edge_classes, node_boxes = augment_pid_sample(
                image, graph, random.Random(f'{self.seed}:{self.epoch}:{idx}:{repeat}'))
        image = torch.from_numpy(np.array(image)).float().div_(255)
        return (
            image[None, None],
            nodes,
            edges,
            self.ids[idx],
            node_classes,
            edge_classes,
            node_boxes,
        )


def resolve_cache_dir(cache_dir=None):
    return Path(cache_dir) if cache_dir else Path.home() / ".cache" / "relationformer"


def augment_pid_sample(image, graph, rng):
    """Paired affine image/box/edge transform, with clipping and new border nodes."""
    from PIL import ImageEnhance, ImageFilter
    from pid_graph import transform_graph, crop_graph
    width,height=image.size
    nodes,edges,classes,edge_classes,boxes=graph
    names={v:k for k,v in PID_NODE_CLASS_TO_ID.items()}
    edge_names={v:k for k,v in PID_EDGE_CLASS_TO_ID.items()}
    raw=[]
    for i,(b,c) in enumerate(zip(boxes.tolist(),classes.tolist())):
        x,y,w,h=b
        raw.append(dict(id=str(i),label=names[c],xmin=(x-w/2)*width,ymin=(y-h/2)*height,
                        xmax=(x+w/2)*width,ymax=(y+h/2)*height))
    es=[dict(source=str(u),target=str(v),label=edge_names[c]) for (u,v),c in zip(edges.tolist(),edge_classes.tolist())]
    angle=np.deg2rad(rng.choice([0,90,180,270])+rng.uniform(-4,4))
    scale=rng.uniform(.9,1.1)
    linear=scale*np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    linear=linear@np.diag([rng.choice([-1,1]),rng.choice([-1,1])])
    origin=np.array([width/2,height/2])
    matrix=np.column_stack((linear,origin-linear@origin))
    inverse=np.linalg.inv(np.vstack((matrix,[0,0,1])))[:2]
    image=image.transform(image.size,Image.Transform.AFFINE,inverse.ravel(),Image.Resampling.BILINEAR,fillcolor=255)
    raw,es=transform_graph(raw,es,matrix)
    raw,es=crop_graph(raw,es,[0,0,width,height],border_size=8*width/1500)
    mapping={n['id']:i for i,n in enumerate(raw)}
    bs=torch.tensor([[(n['xmin']+n['xmax'])/(2*width),(n['ymin']+n['ymax'])/(2*height),
                      (n['xmax']-n['xmin'])/width,(n['ymax']-n['ymin'])/height] for n in raw],dtype=torch.float32).reshape(-1,4)
    image=ImageEnhance.Brightness(image).enhance(rng.uniform(.9,1.1))
    image=ImageEnhance.Contrast(image).enhance(rng.uniform(.9,1.1))
    if rng.random()<.25:
        image=image.filter(ImageFilter.GaussianBlur(rng.uniform(.1,.6)))
    return (image,bs[:,:2],torch.tensor([(mapping[e['source']],mapping[e['target']]) for e in es],dtype=torch.long).reshape(-1,2),
            torch.tensor([PID_NODE_CLASS_TO_ID[n['label']] for n in raw],dtype=torch.long),
            torch.tensor([PID_EDGE_CLASS_TO_ID[e['label']] for e in es],dtype=torch.long),bs)


_WORKER_IMAGES = None
_WORKER_IMAGE_SIZE = None


def _init_pid_worker(images_path, image_size):
    global _WORKER_IMAGES, _WORKER_IMAGE_SIZE
    _WORKER_IMAGES = np.load(images_path, mmap_mode="r+")
    _WORKER_IMAGE_SIZE = image_size


def _preprocess_pid_sample(job):
    index, image_path, graph_path = job
    with Image.open(image_path) as image:
        width, height = image.size
        resized = image.convert("L").resize(_WORKER_IMAGE_SIZE, Image.BILINEAR)
        _WORKER_IMAGES[index] = np.asarray(resized, dtype=np.uint8)
    nodes, edges, node_classes, edge_classes, node_boxes = load_pid_graph(
        graph_path, width, height
    )
    return index, nodes, edges, node_classes, edge_classes, node_boxes


def preprocess_pid_samples(samples, image_size, cache_prefix, num_workers=None):
    """Decodes/resizes every image into a uint8 memmap and parses graphs to tensors, once.

    Returns (images memmap [N, H, W], list of (nodes, edges)). The graphs pickle is written
    last and doubles as the completion marker for the image memmap.
    """
    images_path = Path(f"{cache_prefix}_images.npy")
    graphs_path = Path(f"{cache_prefix}_graphs.pickle")

    if images_path.is_file() and graphs_path.is_file():
        with open(graphs_path, "rb") as graphs_file:
            graphs = pickle.load(graphs_file)
        images = np.load(images_path, mmap_mode="r")
        cache_is_compatible = all(
            isinstance(sample, tuple) and len(sample) == 5 for sample in graphs
        )
        if images.shape[0] == len(graphs) == len(samples) and cache_is_compatible:
            print(f"Loaded preprocessed P&ID samples from {cache_prefix}_*")
            return images, graphs

    print(f"Preprocessing {len(samples)} P&ID samples into {cache_prefix}_* (first run)...")
    start_time = time.time()
    images_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = image_size
    images = np.lib.format.open_memmap(
        images_path, mode="w+", dtype=np.uint8, shape=(len(samples), height, width)
    )
    images.flush()
    del images

    jobs = [
        (index, str(image_path), str(graph_path))
        for index, (image_path, graph_path, _) in enumerate(samples)
    ]
    graphs = [None] * len(samples)
    num_workers = num_workers or os.cpu_count() or 1
    with Pool(
        num_workers, initializer=_init_pid_worker, initargs=(str(images_path), tuple(image_size))
    ) as pool:
        for done, (index, nodes, edges, node_classes, edge_classes, node_boxes) in enumerate(
            pool.imap_unordered(_preprocess_pid_sample, jobs, chunksize=64), 1
        ):
            graphs[index] = (nodes, edges, node_classes, edge_classes, node_boxes)
            if done % 5000 == 0:
                print(f"  {done}/{len(samples)} samples preprocessed ({time.time() - start_time:.0f}s)")

    with open(graphs_path, "wb") as graphs_file:
        pickle.dump(graphs, graphs_file)
    print(f"Preprocessing done in {time.time() - start_time:.0f}s")

    return np.load(images_path, mmap_mode="r"), graphs


def index_pid_samples(
    root,
    split_seed,
    max_nodes,
    cache_dir=None,
    train_sources=None,
    test_sources=None,
    validation_percent=5,
    source_limits=None,
):
    """Validate pairs, cache an index and report, and split by source drawing."""
    root = Path(root)
    cache_dir = resolve_cache_dir(cache_dir)
    train_sources = set(train_sources or [])
    test_sources = set(test_sources or [])
    if train_sources & test_sources:
        overlap = sorted(train_sources & test_sources)
        raise ValueError(f"P&ID sources cannot be both train and test: {overlap}")
    if not 0 < validation_percent < 100:
        raise ValueError("validation_percent must be between 1 and 99")
    # Explicit source roots also support Kaggle's symlinked read-only datasets.
    graph_paths = sorted(g for source in root.iterdir() if source.is_dir() for g in source.rglob('*.graphml'))
    fingerprint = [(str(g.relative_to(root)),g.stat().st_size,g.stat().st_mtime_ns,
                    g.with_suffix('.png').stat().st_mtime_ns if g.with_suffix('.png').is_file() else None)
                   for g in graph_paths]
    cache_key = hashlib.sha1(
        f"{root.resolve()}:{split_seed}:{max_nodes}:{sorted(train_sources)}:"
        f"{sorted(test_sources)}:{source_limits}:{fingerprint}:{validation_percent}:{PID_CACHE_SCHEMA_VERSION}".encode()
    ).hexdigest()[:16]
    cache_path = cache_dir / f"pid_index_{cache_key}.pickle"

    if cache_path.is_file():
        with open(cache_path, "rb") as cache_file:
            cached = pickle.load(cache_file)
        print(f"Loaded P&ID sample index from {cache_path}")
        return {
            split: [(root / img, root / graph, sample_id) for img, graph, sample_id in samples]
            for split, samples in cached.items()
        }

    print(f"Indexing P&ID samples under {root} (first run, this can take a few minutes)...")
    start_time = time.time()
    if root.name == 'Complete' or any('Complete' in g.relative_to(root).parts for g in graph_paths):
        raise ValueError('Training requires a patch root, not Complete drawings')
    groups = {}
    for graph_path in graph_paths:
        rel = graph_path.relative_to(root)
        if len(rel.parts)<3:
            raise ValueError(f'Expected Source/Drawing/patch.graphml: {rel}')
        groups.setdefault(rel.parts[0],set()).add('/'.join(rel.parts[:2]))
    group_split = {}
    for source, drawings in groups.items():
        ordered=sorted(drawings,key=lambda d: hashlib.sha1(f'{split_seed}:{d}'.encode()).hexdigest())
        if source_limits and source in source_limits:
            if len(ordered)<source_limits[source]:
                raise ValueError(f'{source}: need {source_limits[source]} drawings, found {len(ordered)}')
            ordered=ordered[:source_limits[source]]
        if source in test_sources:
            group_split.update({d:'test' for d in ordered})
        elif not train_sources or source in train_sources:
            if len(ordered)<2:
                raise ValueError(f'{source}: at least two drawings required for disjoint train/validation')
            count=max(1,round(len(ordered)*validation_percent/100))
            group_split.update({d:('valid' if i<count else 'train') for i,d in enumerate(ordered)})
    print(f"Found {len(graph_paths)} GraphML files, filtering...")

    splits = {split: [] for split in PatchedPIDDataset.SPLITS}
    report = dict(node_classes=Counter(),edge_classes=Counter(),empty_graphs=0,edgeless_graphs=0,
                  max_nodes=0,boxes_outside=0,drawings={s:len(v) for s,v in groups.items()})
    for index, graph_path in enumerate(graph_paths, 1):
        image_path = graph_path.with_suffix(".png")
        if not image_path.is_file():
            raise FileNotFoundError(f'Missing image paired with {graph_path}')

        relative_graph = graph_path.relative_to(root)
        source = relative_graph.parts[0]
        split = group_split.get('/'.join(relative_graph.parts[:2]))
        if split is None:
            continue

        ns,es=parse_graph(graph_path)
        if max_nodes is not None and len(ns)>max_nodes:
            raise ValueError(f'{graph_path}: {len(ns)} nodes exceed {max_nodes} queries')
        report['node_classes'].update(n['label'] for n in ns)
        report['edge_classes'].update(e['label'] for e in es)
        report['empty_graphs']+=not ns
        report['edgeless_graphs']+=not es
        report['max_nodes']=max(report['max_nodes'],len(ns))
        with Image.open(image_path) as im:
            width,height=im.size
        report['boxes_outside']+=sum(n['xmin']<0 or n['ymin']<0 or n['xmax']>width or n['ymax']>height for n in ns)
        splits[split].append((image_path.relative_to(root).as_posix(),
                             relative_graph.as_posix(),relative_graph.with_suffix('').as_posix()))

        if index % 5000 == 0:
            print(f"  {index}/{len(graph_paths)} graphs processed ({time.time() - start_time:.0f}s)")

    print(
        f"Indexing done in {time.time() - start_time:.0f}s "
        + ", ".join(f"{split}={len(samples)}" for split, samples in splits.items())
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as cache_file:
        pickle.dump(splits, cache_file)
    report['splits']={split:len(samples) for split,samples in splits.items()}
    cache_path.with_suffix('.json').write_text(json.dumps(report,indent=2))
    print(f'Dataset statistics: {cache_path.with_suffix(".json")}')

    return {
        split: [(root / img, root / graph, sample_id) for img, graph, sample_id in samples]
        for split, samples in splits.items()
    }


def load_pid_graph(graph_path, width, height):
    """Read GraphML, clip boxes and retain isolated/edgeless targets."""
    raw_nodes, raw_edges = parse_graph(Path(graph_path))
    nodes, boxes, classes, mapping = [], [], [], {}
    for n in raw_nodes:
        x1, y1 = max(0., n["xmin"]), max(0., n["ymin"])
        x2, y2 = min(float(width), n["xmax"]), min(float(height), n["ymax"])
        if x2 <= x1 or y2 <= y1:
            continue
        mapping[n["id"]] = len(nodes)
        center = ((x1+x2)/(2*width), (y1+y2)/(2*height))
        nodes.append(center)
        boxes.append((*center, (x2-x1)/width, (y2-y1)/height))
        classes.append(PID_NODE_CLASS_TO_ID[n["label"]])
    edges = [e for e in raw_edges if e["source"] in mapping and e["target"] in mapping]
    return (
        torch.tensor(nodes, dtype=torch.float32).reshape(-1, 2),
        torch.tensor([(mapping[e["source"]], mapping[e["target"]]) for e in edges],
                     dtype=torch.long).reshape(-1, 2),
        torch.tensor(classes, dtype=torch.long),
        torch.tensor([PID_EDGE_CLASS_TO_ID[e["label"]] for e in edges], dtype=torch.long),
        torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
    )


def is_trainable_pid_graph(graph_path, max_nodes=None):
    nodes, _ = parse_graph(Path(graph_path))
    if max_nodes is not None and len(nodes) > max_nodes:
        raise ValueError(f"{graph_path}: {len(nodes)} nodes exceed {max_nodes} object queries")
    return True


def image_graph_collate_road_network(batch):
    images = torch.cat([item[0] for item in batch], 0).contiguous()
    nodes = [item[1] for item in batch]
    edges = [item[2] for item in batch]
    ids = [item[3] for item in batch]
    if len(batch[0]) >= 6:
        node_classes = [item[4] for item in batch]
        edge_classes = [item[5] for item in batch]
        if len(batch[0]) >= 7:
            node_boxes = [item[6] for item in batch]
            return [images, nodes, edges, ids, node_classes, edge_classes, node_boxes]
        return [images, nodes, edges, ids, node_classes, edge_classes]
    return [images, nodes, edges, ids]


def load_dataset(dataset_path):
    """
    Loads the chosen split of the data

    :param dataset_path: path of the data split pickle
    :param max_prev_node: only return the last previous 'max_prev_node' elements in the adjacency row of a node
    :param return_coordinates: returns coordinates on the real map for each datapoint
    :return:
    """
    with open(dataset_path, "rb") as pickled_file:
        dataset = pickle.load(pickled_file)

    list_nodes = []
    list_edges = []
    ids = list(dataset.keys())
    random.Random(42).shuffle(
        ids
    )  # permute to remove any correlation between consecutive datapoints

    for id in ids:
        datapoint = dataset[id]

        # Retrieve from dataset
        nodes = (torch.FloatTensor(datapoint["nodes"]) + 1) / 2
        edges = torch.tensor((datapoint["edges"]))[:, :2]

        # Sort edges
        edges_sorted = torch.sort(edges, 1)[0]
        edges_sorted = torch.stack(sorted(edges_sorted, key=lambda a: (a[0], a[1])))

        # Get rid of duplicate edges and nodes that do not participate in edges
        edges_clean = torch.unique(edges_sorted, dim=0)
        participating_nodes = torch.unique(edges_clean)

        for node in torch.arange(nodes.shape[0] - 1, -1, -1):
            if node not in participating_nodes:
                edges_clean[edges_clean > node] -= 1

        nodes_clean = nodes[participating_nodes]

        list_nodes.append(nodes_clean)
        list_edges.append(edges_clean)

    return ids, list_nodes, list_edges


def load_images(ids, images_path):
    """
    Load images from arrays in pickle files

    :param ids: ids of the images in the data order
    :param images_path: path of the pickle file
    :return: the images, as pytorch tensors
    """
    images = []
    with open(images_path, "rb") as pickled_file:
        images_features = pickle.load(pickled_file)
    for id in ids:
        img = torch.FloatTensor(images_features["{:0>7d}".format(int(id))])
        assert img.shape[1] == img.shape[2]
        assert img.shape[1] in {64}
        images.append(img)

    return images


def load_raw_images(ids, images_path):
    """
    Load images from raw files

    :param ids: ids of the images in the data order
    :param images_path: path of the raw images
    :return: the images, as pytorch tensors
    """
    images = []
    for count, id in enumerate(ids):
        # if count % 10000 == 0:
        #     print(count)
        image_path = images_path + "{:0>7d}".format(int(id)) + ".png"
        img = Image.open(image_path).convert("L")
        img = tvf.to_tensor(img)
        assert img.shape[1] == img.shape[2]
        assert img.shape[1] in {64, 128}
        images.append(img)
    return images


def build_road_network_data(config, mode="split"):
    if config.DATA.DATASET == "patched-pid-2D":
        dataset_kwargs = {
            "root_path": os.environ.get("PID2GRAPH_DATA_PATH", config.DATA.DATA_PATH),
            "image_size": config.DATA.IMG_SIZE,
            "split_seed": config.DATA.SEED,
            "max_nodes": config.MODEL.DECODER.OBJ_TOKEN,
            "cache_dir": os.environ.get(
                "RELATIONFORMER_CACHE_DIR", getattr(config.DATA, "CACHE_DIR", None)
            ),
            "train_sources": getattr(config.DATA, "TRAIN_SOURCES", None),
            "test_sources": getattr(config.DATA, "TEST_SOURCES", None),
            "validation_percent": getattr(config.DATA, "VALIDATION_PERCENT", 5),
            "source_limits": vars(config.DATA.SOURCE_LIMITS) if getattr(config.DATA, 'SOURCE_LIMITS', None) else None,
            "augment": getattr(config.DATA, 'AUGMENT', False),
            "cache_images": getattr(config.DATA, 'CACHE_IMAGES', False),
        }
        if mode == "split":
            return PatchedPIDDataset(split="train", **dataset_kwargs), PatchedPIDDataset(
                split="valid", **dataset_kwargs
            )
        if mode == "test":
            return PatchedPIDDataset(split="test", **dataset_kwargs)
        raise ValueError(f"Unsupported data build mode: {mode}")

    if mode == "split":
        train_ds = ToulouseRoadNetworkDataset(root_path=config.DATA.DATA_PATH, split="train")
        val_ds = ToulouseRoadNetworkDataset(root_path=config.DATA.DATA_PATH, split="valid")

        return train_ds, val_ds
    elif mode == "test":
        test_ds = ToulouseRoadNetworkDataset(root_path=config.DATA.DATA_PATH, split="test")
        return test_ds


if __name__ == "__main__":
    import cv2
    import numpy as np
    from torch.utils.data import DataLoader
    from torch.nn.utils.rnn import pad_sequence

    def custom_collate_fn(batch):
        """
        Custom collate function ordering the element in a batch by descending length

        :param batch: batch from pytorch dataloader
        :return: the ordered batch
        """
        x_adj, x_coord, y_adj, y_coord, img, seq_len, ids = zip(*batch)

        x_adj = pad_sequence(x_adj, batch_first=True, padding_value=0)
        x_coord = pad_sequence(x_coord, batch_first=True, padding_value=0)
        y_adj = pad_sequence(y_adj, batch_first=True, padding_value=0)
        y_coord = pad_sequence(y_coord, batch_first=True, padding_value=0)
        img, seq_len = torch.stack(img), torch.stack(seq_len)

        seq_len, perm_index = seq_len.sort(0, descending=True)
        x_adj = x_adj[perm_index]
        x_coord = x_coord[perm_index]
        y_adj = y_adj[perm_index]
        y_coord = y_coord[perm_index]
        img = img[perm_index]
        ids = [ids[perm_index[i]] for i in range(perm_index.shape[0])]

        return x_adj, x_coord, y_adj, y_coord, img, seq_len, ids

    class obj:
        def __init__(self, dict1):
            self.__dict__.update(dict1)

    def dict2obj(dict1):
        return json.loads(json.dumps(dict1), object_hook=obj)

    config = "configs/road_2D_deform_detr.yaml"
    with open(config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2obj(config)

    train_ds, val_ds = build_road_network_data(config, mode="split")
    # dataloader = DataLoader(train_ds, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)

    for i in [14, 2, 4, 6, 30, 26, 43, 24, 69173, 48360, 60201]:
        ret = train_ds[i]  # some strange cases 14, 2, 4, 6, 30, 26, 43, 24
        print(ret[-1])

        nodes_pixels = (ret[1] * ret[0].shape[-1]).type(torch.int32).numpy()

        image = ret[0].squeeze().cpu().numpy()
        image = np.flip(image, 0).copy()

        for node in nodes_pixels:
            image = cv2.circle(image, node, 5, (0, 0, 0), 2)
            cv2.imshow("testing", image)
            cv2.waitKey()

        print(ret[-2])
