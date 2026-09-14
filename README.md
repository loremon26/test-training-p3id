# Relationformer per P&ID

Implementazione della pipeline Relationformer di *From Engineering Diagrams to Graphs: Digitizing P&IDs with Transformers*, arXiv:2411.13929v3. Il PDF è nella repo. Il metodo Modular Digitization non è utilizzato.

La pipeline usa Dataset-P&ID per il pretraining, allena detection e relazioni, ricompone il grafo completo ed esporta JSON, GraphML e immagini annotate. Le linee sono **archi solid/non-solid tra nodi**, non maschere di segmentazione o OCR. Gli archi sono non orientati: la randomizzazione durante il training non insegna il verso del flusso.

## Dati disponibili e limiti

`archive` è Dataset-P&ID del paper locale 2109.03794v1: 500 disegni sintetici, 30.000 patch JPEG 1280×1280 e 195.759 box YOLO su 32 classi. I file YOLO non annotano la connettività. La stessa collezione è già convertita in grafi sotto `PID2Graph/Complete/Dataset PID` e `PID2Graph/Patched/Dataset PID`; il training usa quest'ultima, senza duplicare un converter. Lo split train/val incluso nell'archivio non viene usato: è per patch e 498 disegni compaiono in entrambi i lati. Il loader crea invece uno split 95:5 per disegno.

Questa è una sostituzione pratica, non equivalente a Synthetic 700: offre 500 anziché circa 2.000 disegni e i GraphML contengono general, valve, instrumentation, arrow e nodi strutturali, ma nessun esempio tank, pump o inlet/outlet. Le augmentation online aumentano la variabilità senza inventare nuove classi. Poiché Dataset PID entra nel training, non è più un benchmark indipendente; restano esclusi dal training `PID2Graph OPEN100` e `PID2Graph Synthetic`.

Synthetic 700 è il nome del dataset generato usando circa 700 template, non una raccolta di 700 P&ID. Il generatore locale precedente resta disponibile come alternativa (`--synthetic-source Generated`) quando servono layout aggiuntivi o copertura preliminare delle classi mancanti. Usa 225 template pubblici più 24 procedurali.

I fogli pubblici sono CC BY-SA 4.0: la cartella template conserva README originale, autore, revisione Git, hash dei fogli e licenza nel manifest. I 24 template procedurali aggiunti sono offerti come CC0. Conservare attribuzione e condizioni CC BY-SA per gli asset derivati quando si distribuiscono i sintetici. Nessun dato proprietario viene scaricato.

Per il fine tuning occorrono ancora **60 P&ID reali annotati**, forniti separatamente. La pipeline li verifica e si ferma se mancano. Le prestazioni del paper, soprattutto su OPEN100, non sono garantite usando solo sintetici.

## Installazione

Python >=3.10. Su Kaggle mantenere torch, torchvision e CUDA preinstallati:

```bash
pip install -r requirements-kaggle.txt
```

In locale installare la coppia torch/torchvision adatta al driver, poi:

```bash
pip install -r requirements.txt
```

CairoSVG serve per estrarre i template pubblici; richiede la libreria di sistema Cairo (su Linux: libcairo2). Per lavorare offline, preparare e trasferire la cartella dei template. La rete usa l'implementazione PyTorch della deformable attention già presente in models/ops: non richiede compilazione CUDA, ma può essere più lenta dell'operatore compilato.

## Pretraining con Dataset-P&ID

Non serve rigenerare né ripatchare i file in `archive`: i GraphML necessari al Relationformer sono già abbinati alle immagini in `PID2Graph/Patched/Dataset PID`.

```bash
export RELATIONFORMER_CACHE_DIR=/tmp/relationformer-cache
python train.py --phase pretrain --config configs/road_2D.yaml \
  --data-root PID2Graph/Patched --synthetic-source "Dataset PID" \
  --output-dir trained_weights/pretrain
```

`Dataset PID` non va incluso nella successiva valutazione. Per una valutazione indipendente usare OPEN100 e PID2Graph Synthetic.

## Generatore alternativo

```bash
python scripts/generate_synthetic_pid.py \
  --prepare-templates --templates data/templates \
  --output-dir data/generated/Complete --count 2000 --seed 10

python scripts/patch_pid.py \
  --input-dir data/generated/Complete \
  --output-dir data/training/Patched/Generated \
  --patch-size 1500 --stride 750 --resize 7000 4500
```

Il generatore dispone i simboli in celle con jitter, costruisce un albero connettente più due collegamenti, instrada tratti ortogonali evitando le altre apparecchiature, divide le intersezioni e consolida segmenti sovrapposti. Introduce crossing e ankle e verifica la connessione del grafo. Le classi e le dimensioni variano secondo distribuzioni dichiarate nel codice; il default di 42 simboli dà circa 130 nodi totali, in linea con l'ordine di grandezza della tabella A3. Non è un simulatore di processo industriale.

Per un primo controllo usare --count 2 e una cartella di output nuova. Il manifest registra seed effettivo (anche se un layout viene rigenerato), parametri, classi, conteggi e percorsi. Gli output esistenti non vengono sovrascritti dal patcher; dopo un'interruzione del preprocessing usare una nuova cartella per il disegno incompleto. Per nuovi simboli si può fornire una cartella PNG e un templates.json con lo stesso schema, dopo aver verificato classi e licenza.

Il patcher interpreta 4500×7000 come **altezza × larghezza**; --resize usa invece **larghezza altezza**. Il resize è anisotropico, senza mantenere l'aspect ratio. I box subiscono la stessa trasformazione. Le parti visibili dei simboli sono mantenute; i tratti che attraversano la patch senza endpoint interno producono due border node. Le patch vuote e senza archi sono ammesse.

Ogni disegno produce 45 patch con questi default; il numero varia con geometria, stride e dimensione. Con 2.000 disegni si ottengono 90.000 patch prima dell'augmentation online. Questo non ricrea i 170.944/8.998 campioni offline del paper e comporta un diverso numero di aggiornamenti per epoca. Non duplicare validation o benchmark per raggiungere artificialmente quei conteggi.

Per un singolo disegno:

```bash
python scripts/patch_pid.py --image plan.png --graph plan.graphml \
  --output-dir data/patches/plan --patch-size 1500 --stride 750
```

Si può omettere --graph per produrre patch di inferenza. Per patch 2000×2000 usare --patch-size 2000, con stride <=1000.

I file manifest.json contengono dimensioni originali e ridimensionate, scala, offset, nomi delle patch e annotazioni. I vecchi *_xy_shifts.npy del benchmark non vengono caricati con pickle.

## Training in due fasi

Struttura richiesta (una cartella per disegno originale):

```text
data/training/Patched/
  Dataset PID/0/0.png + 0.graphml
  Dataset PID/1/...
  Real/real_000/...
  Real/real_001/...
```

Il loader esistente è stato esteso e usa il parser GraphML della utility di preparazione: connector → ankle, border_node → border, inlet/outlet → inlet_outlet, non-solid → non_solid. Legge le chiavi per attr.name, non per posizione d0/d1. Scarta soltanto il background artificiale dei piani Complete, non i nodi isolati. Controlla classi, box, archi pendenti, immagini mancanti e limite di 400 nodi. Box parzialmente esterni vengono clippati; errori strutturali non vengono nascosti.

L'indice produce statistiche JSON nella cache. I manifest di patch sono disponibili in dataset.metadata. La lettura delle immagini è lazy; DATA.CACHE_IMAGES abilita facoltativamente la vecchia cache memmap, molto più voluminosa. RELATIONFORMER_CACHE_DIR imposta una cartella scrivibile.

```bash
export RELATIONFORMER_CACHE_DIR=/tmp/relationformer-cache
python train.py --phase pretrain --config configs/road_2D.yaml \
  --data-root PID2Graph/Patched --synthetic-source "Dataset PID" \
  --output-dir trained_weights/pretrain
```

Default: ResNet-101 ImageNet, hidden 256, 400 object + 1 relation token, 11 classi nodo incluso no-object e 3 classi arco incluso no-edge. Class head lineare, box head MLP e relation head MLP sono riutilizzati. Il matching di training ora include box L1/gIoU oltre a classi e centri; gli archi sono rimappati sugli indici del matching. La cardinality loss è una metrica senza gradiente, come nell'implementazione ereditata.

Su Kaggle il notebook usa entrambe le T4 tramite `torch.nn.DataParallel`, con microbatch 2 (suddiviso tra le GPU) e accumulo fino a **batch effettivo 20**, AMP su CUDA, 80 epoche, AdamW, LR 1e-4, backbone LR 3e-5, loss box/class/card/node/edge 2/2/1/4/3. Se manca memoria usare --batch-size 1 mantenendo --effective-batch-size 20. L'ultimo gruppo dell'epoca può contenere meno di 20 campioni. La normalizzazione delle loss nei microbatch rende l'accumulo un'approssimazione del batch fisico 20. Per una sola GPU usare `--cuda_visible_device 0`.

Backbone e proiezioni delle feature restano in FP32: FrozenBatchNorm con pesi casuali può causare overflow fp16 nei layer profondi del ResNet. Transformer e teste usano AMP; GradScaler può saltare i primi aggiornamenti mentre regola la scala. Questa scelta di precisione è stata verificata anche su GPU, ma non è specificata dal paper.

Lo split 95:5 è deterministico **per disegno**, separatamente per sorgente, prima delle augmentation. Con pochi disegni viene riservato almeno uno alla validation: due disegni producono quindi uno split 1:1, adatto solo al test. Sono applicate online rotazioni di 90° e ±4°, flip, scala 0.9–1.1, brightness/contrast 0.9–1.1 e blur. Box e archi vengono trasformati insieme, con nuovi border sui tagli. La validation non ha augmentation casuali.

Preparare i reali e avviare il fine tuning:

```bash
python scripts/patch_pid.py --input-dir /path/to/real/Complete \
  --output-dir data/training/Patched/Real

python train.py --phase finetune --config configs/road_2D.yaml \
  --data-root data/training/Patched --output-dir trained_weights/finetune \
  --pretrained trained_weights/pretrain/best.pt \
  --synthetic-source "Dataset PID" \
  --finetune-synthetic-drawings 500 --real-drawings 60
```

--pretrained trasferisce solo i pesi. Il fine tuning usa i 500 Dataset-P&ID e 60 reali con TRAIN.REAL_WEIGHT=3: ogni patch reale ha tre augmentation per epoca rispetto a una sintetica. La percentuale reale effettiva dipende dai conteggi delle patch: non è forzata al 37%.

Early stopping e best.pt sono basati sulla **validation loss totale**. Pretraining: nessun early stopping; fine tuning: patience 10, min_delta 0, entrambi configurabili. Val loss e metriche sono calcolate a ogni epoca; preview predizione/ground truth sono salvate in validation/epoch_XXX.

Output: last.pt, best.pt, config.yaml/json, split.json, history.jsonl e train.log. I checkpoint completi contengono rete, optimizer, scheduler, scaler, epoca, posizione nell'epoca, contatore di step, best loss/patience, RNG e firma dei dati. La firma verifica ID, dimensioni dei file immagine e contenuto GraphML; non è un hash di tutti i pixel. Caricare solo checkpoint fidati: il resume usa la serializzazione PyTorch completa.

```bash
python train.py --phase pretrain --config trained_weights/pretrain/config.yaml \
  --data-root PID2Graph/Patched --synthetic-source "Dataset PID" \
  --output-dir trained_weights/pretrain \
  --resume trained_weights/pretrain/last.pt
```

Il checkpoint intermedio viene scritto ogni 250 aggiornamenti e alla scadenza del budget TRAIN.MAX_HOURS; un'interruzione improvvisa può perdere gli aggiornamenti successivi all'ultimo checkpoint. Per un fine tuning ripreso usare --phase finetune e il relativo last.pt. Non cambiare microbatch, seed o dati in resume. La riproducibilità bit a bit su hardware/CUDA differenti non è garantita. Su CPU usare --device cpu; --no-imagenet serve per test offline senza download dei pesi.

## Inferenza, merge e visualizzazione

```bash
python predict_image.py plan.png --checkpoint trained_weights/finetune/best.pt \
  --output-dir results/plan

python predict_image.py patch.png --single-patch \
  --checkpoint trained_weights/finetune/best.pt --output-dir results/patch
```

L'inferenza completa salva patch/manifest, predizioni di ogni patch, graph.json, graph.graphml e prediction.png. Box e confidenze sono sempre esportati. I colori distinguono simboli e nodi strutturali; gli archi solid sono verdi e i non-solid arancioni tratteggiati.

Per fondere predizioni già disponibili:

```bash
python scripts/merge_pid_graph.py --manifest results/plan/patches/manifest.json \
  --predictions results/plan/predictions --output results/plan_merged
```

I JSON di patch usano box xyxy in pixel della patch e ID locali stringa. Il merge applica c_hat = c - 0.4 exp(-3 abs(2d/S)), dove d è la distanza minima del box dal bordo, limitata a zero per box esterni. Poi usa filtro 0.15, NMS class-aware IoU 0.8, WBF class-aware IoU 0.4, rimappatura degli archi e riconnessione geometrica dei border. I duplicati soppressi conservano una mappa verso il nodo superstite, quindi i loro archi non vengono semplicemente persi.

La riconnessione divide i tratti sovrapposti nei punti border e contrae i border di grado due con archi della stessa classe. La tolleranza predefinita è 8 pixel nelle coordinate ridimensionate. I border non risolti restano nel grafo; self-loop e nodi isolati sono eliminati come nel paper. Soglie e dettagli di riconnessione non sono pubblicati: questa è un'euristica documentata, da validare su dati di validation rappresentativi.

## Valutazione indipendente

```bash
python evaluate_pid.py --dataset-root PID2Graph \
  --checkpoint trained_weights/finetune/best.pt --output-dir results/benchmarks \
  --mode both --sources "PID2Graph OPEN100" "PID2Graph Synthetic"
```

test.py richiama la stessa valutazione; le vecchie metriche road-network non sono usate. --max-samples 1 serve solo per uno smoke test.

Le metriche, in scala 0–1, sono:

- symbol mAP@IoU 0.5 sulle sette classi simbolo;
- node AP@IoU 0.5 ignorando la classe; border esclusi di default, includibili con --include-borders;
- edge mAP: matching Hungarian class-agnostic dei box con costo -gIoU, quindi verifica di endpoint e tipo di arco secondo A1. Nessun filtro IoU è applicato al matching degli archi.

L'AP dei nodi riutilizza l'evaluator esistente con interpolazione COCO a 101 punti, escludendo classi prive di ground truth e senza il precedente limite di 100 detections. L'AP degli archi integra la curva precision/recall ai livelli di confidenza distinti, con i FN nel denominatore: equivale all'integrazione non interpolata di sklearn sulle predizioni osservate, moltiplicata per il recall raggiungibile. Il paper non spiega come traduce i FN in input a sklearn: questa convenzione è esplicita e non implica identità numerica.

Per impostazione predefinita A1 è seguito letteralmente: un arco esistente ma di classe sbagliata è FP; viene aggiunto FN soltanto quando manca la coppia di endpoint. --edge-class-mismatch-fn abilita anche FN per la classe corretta. Le classi senza archi veri sono riportate null ed escluse dalla media. La classificazione no-object/no-edge prevale sulle classi foreground; tra le predizioni rimanenti si usano soglie basse 0.05 per conservare il ranking AP. Le soglie di visualizzazione sono 0.60/0.75.

Patched usa le patch pubbliche così come sono. Stitched riparte da Complete e ricrea patch 1500/750, così scala e offset sono noti e non dipendono da vecchi array pickle. Nell'archivio locale Synthetic ha patch 2000×2000; gli altri due hanno patch 1500×1500 e overlap non uniforme. Perciò queste modalità non devono essere presentate come la stessa identica geometria di valutazione degli autori. Complete serve a questa verifica, non al loader di training.

## Notebook Kaggle

Aprire kaggle_pretrain_relationformer.ipynb e poi kaggle_finetune_relationformer.ipynb. Contengono setup, configurazione dei percorsi, generazione/indicizzazione, controlli GPU/dati, training/resume, curve, preview e download dei checkpoint. Il secondo verifica esplicitamente i 60 reali prima di procedere e include l'evaluation indipendente.

I precedenti kaggle_train_relationformer.ipynb e notebook1960e21366.ipynb sono esperimenti storici, non il protocollo aggiornato. Non usarli per questa pipeline.

## Verifiche

```bash
python scripts/smoke_pid.py --templates data/templates --output-dir data/check_nuovo --device cpu
python scripts/smoke_pid.py --templates data/templates --output-dir data/check_cuda --device cuda --amp
git diff --check
```

Il test genera due disegni, verifica riproducibilità e split, patcha i grafi, controlla gli archi che attraversano una patch con entrambi gli endpoint fuori, prova il merge di una linea lunga, verifica AP=1 sui grafi perfetti e AP=0 per archi mancanti/errati, quindi esegue forward e backward del modello completo a 512×512. Verifica loss su grafi vuoti/isolati, validation deterministica e inferenza→merge. Salva test_report.json e immagini per ispezione.

Verifiche eseguite il 14 settembre 2026: smoke CPU e CUDA AMP (RTX A1000 6 GB), un'epoca minima con interruzione/ripresa e checkpoint best, lettura dei 36.561 GraphML locali, evaluation patched/stitched su un esempio per benchmark e sintassi delle celle dei due notebook. Il nuovo indice Dataset-P&ID produce 18.488 patch train da 475 disegni e 974 patch validation da 25 disegni. L'esecuzione completa dei notebook su Kaggle e l'addestramento 500+60 non sono stati eseguiti. I checkpoint in data/smoke_training sono soltanto artefatti di test, non modelli utili per la digitizzazione.

Nel round-trip di un disegno sintetico completo usando patch ground truth, il merge ha mantenuto symbol/node AP=1 ma edge mAP circa 0.992: NMS/WBF e riconnessione geometrica non garantiscono ricostruzione senza perdita nemmeno con patch perfette. Le predizioni di un modello reale possono richiedere soglie migliori, scelte sulla validation.

Le scelte non determinate dal paper includono: template e generatore, conteggi/augmentation online, seed/split, lettura W×H del resize, box strutturali da 8 pixel, quattro layer encoder/decoder e hidden 256 ereditati, variante del relation-token attention ereditata, matcher box-aware, dettagli delle loss e campionamento negativo (4×, minimo 20), AdamW/ImageNet, weight decay 1e-4, StepLR a epoca 60, clipping 0.1, 80 epoche per fase, patience, soglie e riconnessione, integrazione AP e trattamento delle classi assenti. I risultati vanno misurati, non dedotti dal fatto che la pipeline esegue correttamente.
