# 📊 Riepilogo Dataset — `ai_challenge/data`

Questa cartella contiene **6 file CSV** relativi al trasporto pubblico urbano di **Bari** (AMTAB — Azienda Mobilità e Trasporti Azienda Bari). I dati provengono da un sistema di raccolta OpenData/Elasticsearch e coprono fermate, veicoli, consumi e servizi di mobilità.

---

## 1. `fermate.csv`

| Proprietà         | Valore                   |
|-------------------|--------------------------|
| **Dimensione**    | ~75 KB                   |
| **Righe**         | 1.349 (+ header)         |
| **Colonne**       | 4                        |
| **Encoding**      | UTF-8                    |

### Descrizione
Elenco di tutte le **fermate dei bus AMTAB** sul territorio di Bari, con coordinate geografiche.

### Colonne

| Colonna               | Tipo     | Descrizione                                  |
|-----------------------|----------|----------------------------------------------|
| `idFermata`           | string   | Codice identificativo univoco della fermata  |
| `descrizioneFermata`  | string   | Indirizzo/descrizione della fermata          |
| `latitudine`          | float    | Latitudine (WGS84)                           |
| `longitudine`         | float    | Longitudine (WGS84)                          |

### Esempio di dati
```
03393101, "Viale O. Flacco, di fronte Civ. 4/A", 41.1107, 16.8618
06006001, "Via Alberotanza, 5", 41.0958, 16.8718
05399002, "Via Oberdan INPDAP", 41.1177, 16.8817
```

---

## 2. `postazionibikesharing.csv`

| Proprietà         | Valore                   |
|-------------------|--------------------------|
| **Dimensione**    | ~1.7 KB                  |
| **Righe**         | 32 (+ header)            |
| **Colonne**       | 4                        |
| **Encoding**      | Latin-1                  |

### Descrizione
Elenco delle **stazioni di bike sharing** sul territorio di Bari, con coordinate e numero di biciclette disponibili.

### Colonne

| Colonna          | Tipo   | Descrizione                                       |
|------------------|--------|---------------------------------------------------|
| `Denominazione`  | string | Nome della postazione di bike sharing             |
| `Lat`            | float  | Latitudine (WGS84)                                |
| `Long`           | float  | Longitudine (WGS84)                               |
| `Numero Bici`    | int    | Numero di biciclette disponibili nella postazione |

### Esempio di dati
```
Agraria,         41.1114, 16.8828, 10
Area Sosta Mazzini, 41.1248, 16.8558, 0
Area sosta Rossani, 41.1161, 16.8718, 10
```

---

## 4. `consumi_amtab.csv`

| Proprietà         | Valore                    |
|-------------------|---------------------------|
| **Dimensione**    | ~467 KB                   |
| **Righe**         | 1.000 (+ header)          |
| **Colonne**       | 49                        |
| **Encoding**      | UTF-8                     |

### Descrizione
Dati sui **consumi di carburante dei mezzi AMTAB**, estratti dall'indice Elasticsearch `vw_amtab_consumi`. Ogni riga è un "buono" di rifornimento con informazioni sul veicolo, i chilometri percorsi e il consumo.

### Colonne principali

| Colonna                                         | Descrizione                                      |
|-------------------------------------------------|--------------------------------------------------|
| `hits_hits__source_Data_buono`                  | Data del buono di rifornimento                   |
| `hits_hits__source_Id_buono`                    | ID univoco del buono                             |
| `hits_hits__source_Id_distributore`             | ID del distributore di carburante                |
| `hits_hits__source_Tipo_buono`                  | Tipo buono (es. `P` = pieno)                     |
| `hits_hits__source_Id_bene`                     | ID del mezzo (veicolo)                           |
| `hits_hits__source_Qta_carburante`              | Quantità carburante erogata (litri)              |
| `hits_hits__source_Km_percor`                   | Km percorsi dall'ultimo rifornimento             |
| `hits_hits__source_Km_attuali`                  | Km totali attuali del veicolo                    |
| `hits_hits__source_ConsumoMedioBuonoCorrente`   | Consumo medio (L/km)                             |
| `hits_hits__source_modello`                     | Modello del veicolo (es. "IRISBUS Europolis")    |
| `hits_hits__source_riferimento`                 | Targa/riferimento del veicolo (es. "DE 158 XW") |

### Esempio di dati
- Veicolo **IRISBUS Europolis** (targa DE 158 XW): 5L di carburante, 11 km percorsi, consumo 2.2 L/km
- Veicolo **IRISBUS 491** (targa DB 381 AS): 142L, 312 km percorsi, consumo ≈ 2.197 L/km


---

## 6. `orari_fermate.csv`

| Proprietà         | Valore                    |
|-------------------|---------------------------|
| **Dimensione**    | ~204 KB                   |
| **Righe**         | 1.000 (+ header)          |
| **Colonne**       | 24                        |
| **Encoding**      | UTF-8                     |

### Descrizione
Dati sui **transiti/rilevazioni dei mezzi AMTAB per ora e quartiere**, estratti dall'indice Elasticsearch `vw_amtabmezzi`. Ogni riga rappresenta il passaggio di un veicolo (identificato dalla corsa) in un determinato quartiere/municipio in una data e ora specifici.

### Colonne principali

| Colonna                               | Descrizione                                          |
|---------------------------------------|------------------------------------------------------|
| `hits_hits__source_id_quartiere`      | ID del quartiere di rilevazione                      |
| `hits_hits__source_id_municipio`      | ID del municipio di rilevazione                      |
| `hits_hits__source_id_circoscrizione` | ID della circoscrizione                              |
| `hits_hits__source_ora`               | Ora della rilevazione (es. 15)                       |
| `hits_hits__source_id_corsa`          | Identificativo della corsa (es. "184203")            |
| `hits_hits__source_totale_rilevazioni`| Totale rilevazioni per quella corsa                  |
| `hits_hits__source_id_data`           | Data in formato numerico                             |
| `hits_hits__source_municipio`         | Nome del municipio (es. "MUNICIPIO N.5")             |
| `hits_hits__source_quartiere`         | Nome del quartiere                                   |
| `hits_hits__source_flag_amtab_mezzi`  | Flag AMTAB mezzi                                     |
| `hits_hits__source_data`              | Data della rilevazione (es. "2017-03-07")            |
| `hits_hits__source_anno`              | Anno della rilevazione                               |

### Esempio di dati
- Corsa **184203**, ora 15, Municipio 1, Quartiere PALESE-MACCHIE, 2017-03-07
- Corsa **184536**, ora 15, Municipio 1, Quartiere MURAT, 2017-03-07
- Corsa **185095**, ora 12, Municipio 3, Quartiere MARCONI-SAN GIROLAMO-FESCA, 2017-03-15

---
