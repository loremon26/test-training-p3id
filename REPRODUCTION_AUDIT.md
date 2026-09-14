# Verifica preliminare della riproduzione PID2Graph / Relationformer

Nota del 14 settembre 2026: questo documento conserva l'audit **precedente alle modifiche**.
Per la pipeline implementata, i notebook, i comandi e le approssimazioni aggiornate consultare
[README.md](README.md). Le discrepanze della vecchia repo elencate sotto non descrivono
necessariamente il codice attuale. L'assenza dei dati originali di training resta confermata.

Verifica del 11 settembre 2026 sul PDF locale `2411.13929v3.pdf`, sulla repo
e sull'intero archivio locale PID2Graph. Sono considerate le sezioni III-A,
III-B, III-D, III-E, III-F e le appendici A1, A2 e A5. Modular Digitization
è escluso dall'implementazione richiesta.

## Risultato principale

I tre dataset presenti sono effettivamente utilizzati dagli autori, ma come
**dataset di test**. La tabella A3, pagina 11 del PDF locale, distingue
esplicitamente il loro utilizzo da quello di `Synthetic 700` e `Real World`,
indicati come `Train`. La stessa distinzione compare nelle sezioni III-D e III-F.

`Patched` e `Complete` sono due rappresentazioni degli stessi disegni;
non rappresentano una separazione training/test.

| Raccolta | Ruolo nel paper | Disponibilità locale |
| --- | --- | --- |
| Synthetic 700 | Pretraining e componente sintetica del fine tuning | Assente |
| Real World, 60 P&ID annotati | Componente reale del fine tuning | Assente |
| Dataset-P&ID / Dataset PID | Test | 500 disegni, 19.462 patch |
| PID2Graph OPEN100 | Test reale | 12 disegni, 1.629 patch |
| PID2Graph Synthetic | Test sintetico | 250 disegni, 14.708 patch |

Il controllo dell'indice di `PID2Graph.zip` conferma che non contiene ulteriori
raccolte di training o checkpoint omessi durante l'estrazione. La
[pagina ufficiale del dataset](https://zenodo.org/records/14803338) distribuisce
questo archivio e descrive immagini, grafi e nodi di bordo.

È possibile usare questi dati per un nuovo esperimento, separando i disegni
tra training, validazione e test prima di qualsiasi augmentation. Cambierebbero
però il protocollo e la distribuzione dei dati. Addestrarsi su un benchmark
impedisce di usarlo integralmente come test indipendente per il confronto
con i risultati pubblicati. Il codice della pipeline può essere preparato;
la ripetizione dell'esperimento originale richiede i dati mancanti.

## Procedura di training pubblicata

| Fase | Disegni | Campioni training | Campioni validazione |
| --- | --- | --- | --- |
| Pretraining | 2.000 P&ID da Synthetic 700 | 170.944 | 8.998 |
| Fine tuning | 500 sintetici più 60 reali | 44.019 | 2.317 |

`700` indica il numero di template di simboli, non il numero di disegni.
Il fine tuning mantiene dati sintetici: augmentation dei reali tre volte
quella dei sintetici, con circa 37% di patch reali e 63% sintetiche.
La sezione III-F descrive l'arresto quando la loss si stabilizza o quella
di validazione comincia a crescere.

La tabella A1 riporta batch 20, input 512×512, 80 epoche, learning rate
1e-4, learning rate del backbone 3e-5, ResNet-101, 401 query,
pesi delle loss box/class/card/node/edge pari a 2/2/1/4/3 e
randomizzazione dell'ordine degli estremi degli archi. L'appendice A2
indica split training/validazione 95:5 e una Quadro RTX 8000 da 48 GB.

Il paper non specifica tutti i dettagli necessari per un'identità sperimentale:
seed e liste degli split; intervalli/probabilità delle augmentation; patience
e min_delta dell'early stopping; configurazione completa di ottimizzatore e
scheduler; dettagli completi di loss, matching e campionamento degli archi;
configurazione completa del transformer e inizializzazione dei pesi.
Le 80 epoche sono riportate in una tabella unica, senza una configurazione
separata per ogni fase. Eventuali scelte implementative su questi punti
devono essere documentate come assunzioni, non attribuite agli autori.

## Controllo delle annotazioni locali

Sono stati letti tutti i 36.561 GraphML e controllati abbinamenti e dimensioni
delle immagini tramite i loro header: 35.799 patch e 762 disegni completi.
Non mancano immagini o GraphML abbinati. Non sono stati riscontrati errori di
parsing XML, estremi di archi inesistenti, self-loop o attributi di bounding
box mancanti. Questo controllo non equivale a una verifica visiva di tutte
le annotazioni né a una decodifica integrale di tutti i pixel.

| Fonte delle patch | Dimensione effettiva | Grafi senza archi | Box oltre il bordo |
| --- | --- | --- | --- |
| Dataset PID | 1500×1500 | 50 | 1.569 |
| PID2Graph OPEN100 | 1500×1500 | 44 | 245 |
| PID2Graph Synthetic | 2000×2000 | 0 | 1.811 |

I box oltre il bordo richiedono una gestione esplicita e coerente; non sono
automaticamente annotazioni errate. Non vanno eliminati in silenzio i grafi
senza archi dalla valutazione: sono presenti anche nel benchmark OPEN100.
Nessuna patch supera 400 nodi; il massimo osservato è 188.

Le annotazioni contengono classi dei simboli, bounding box, collegamenti e
classi degli archi. `connector` e `border_node` richiedono la mappatura alle
classi strutturali; nei GraphML esaminati `connector` e `crossing` hanno box
8×8. I disegni completi Dataset PID e OPEN100 includono inoltre nodi
`background`, da interpretare separatamente dalle classi degli oggetti.

`inlet/outlet` compare soltanto in OPEN100. Allenando sulle altre due raccolte
e lasciando OPEN100 interamente al test non ci sarebbero esempi di training
per questa classe. OPEN100 ha soltanto archi solidi. PID2Graph Synthetic
non contiene esempi di instrumentation, presenti invece in Dataset PID.

Sono presenti 762 file `*_xy_shifts.npy`, uno per disegno, da considerare
per il merge delle patch distribuite. Tre campioni ispezionati contengono
ID e traslazioni delle patch; sono array NumPy di oggetti. Le traslazioni
non vanno sostituite con una griglia ricostruita assumendo un overlap unico.

## Preprocessing, merge e valutazione

III-A descrive il ridimensionamento del piano a 4500×7000 e overlap almeno
50%. Nel paper è discusso un formato base di patch 1500×1500 e uno studio
su OPEN100 a 2000×2000. L'archivio sintetico locale contiene invece già
patch 2000×2000: questa differenza va registrata e verificata prima di
presentare risultati come una replica esatta.

III-B richiede simboli più nodi strutturali per gomiti, incroci e intersezioni
delle linee con il bordo della patch. L'output atteso è un grafo con bounding
box e classi dei nodi e classi delle connessioni, non una maschera pixel per
pixel delle tubazioni. Le sette classi di simboli e le tre strutturali
richiedono 10 classi di oggetti più no-object; gli archi richiedono solid,
non-solid e assenza di relazione. Le query sono 400 oggetti più una relazione.

Il merge descritto abbassa la confidenza dei box vicini ai bordi con
`c_hat = c - 0.4 * exp(-3 * abs(2*d/S))`, filtra le basse confidenze,
riporta i box nelle coordinate globali, applica NMS con IoU alto e WBF con
IoU inferiore, quindi rimuove self-loop e nodi isolati. I valori delle soglie
di confidenza, NMS e WBF e i dettagli di riconnessione dei nodi di bordo
non sono specificati completamente. Un merge dei soli rettangoli non basta:
serve rimappare anche gli archi sui nodi risultanti.

`Complete` può essere escluso dal loader di training su patch. Serve però
per verificare il patching e valutare il grafo dell'intero disegno dopo il
merge, compreso il ritorno alle coordinate originali.

III-E e A1 richiedono mAP dei simboli a IoU 0,5, AP dei nodi senza distinzione
di classe e mAP degli archi tramite matching ungherese dei nodi basato su gIoU.
Le metriche geometriche per reti stradali non sostituiscono queste misure.

## Differenze già individuate nella repo

- `configs/road_2D.yaml` usa Dataset PID e PID2Graph Synthetic per il training:
  è già un esperimento sui dati disponibili, come avverte anche il README.
- Il batch è 8 per GPU, quindi 16 su due GPU, anziché 20.
- `trainer.py` ed `evaluator.py` selezionano/arrestano tramite `val_smd`,
  senza calcolare la loss di validazione richiesta per quel criterio.
- Il loader `PatchedPIDDataset` ridimensiona le immagini e carica i grafi,
  ma non applica le augmentation elencate nel paper.
- Manca una gestione esplicita delle due fasi. `--resume` ripristina anche
  ottimizzatore e scheduler; non equivale da solo a un fine tuning separato.
- `prepare_pid2graph_paper_dataset.py` normalizza GraphML già patchati;
  non implementa il patching di un disegno completo.
- `predict_image.py` riduce l'intera immagine a 512×512 ed esporta centri e
  connessioni, senza esportare box, classi e confidenze; manca il merge.
- `test.py` valuta gli archi come rettangoli geometrici; non implementa
  la metrica degli archi dell'algoritmo A1.

ResNet-101, numero di query e teste per bounding box, classi degli oggetti
e classi delle relazioni sono già presenti: sono componenti da riutilizzare.
Su Kaggle occorrerebbero batch effettivo 20, eventuale accumulo dei gradienti,
checkpoint riprendibili e due notebook con passaggio dei pesi tra le fasi.
Compatibilità e consumi GPU devono essere misurati, non sono stati verificati
da questa analisi statica.

Non sono stati modificati gli script, l'architettura o il dataset. Questo
documento conserva la verifica richiesta prima dell'implementazione.
La scelta ancora necessaria è l'accesso ai dati di training originali oppure
l'accettazione di un esperimento con dati/split diversi dal paper.
