# Piano sperimentale

## Esperimenti preliminari

Test per valutare le metriche prodotte da training e test sullo stesso tipo di dataset, per ogni modello (2D e 3D) e per ogni soglia di coerenza (60-90-120).

### Modelli 2D e coerenza 60-90-120

#### 2DP0

- [x] Testare movimento implicito circolare (Alessandro IN TEORIA)

#### 2DP1

- [x] Testare movimento implicito traslazionale (Matteo)

#### 2DP2

- [ ] Testare movimento implicito traslazionale e circolare

### Modelli 3D e coerenza 60-90-120

#### 3DP0

- [ ] Testare movimento implicito circolare

#### 3DP1

- [ ] Testare movimento implicito traslazionale

#### 3DP2

- [ ] Testare movimento implicito traslazionale e circolare

#### 3DP3

- [ ] Testare movimento esplicito circolare

#### 3DP4

- [ ] Testare movimento esplicito traslazionale

#### 3DP5

- [ ] Testare movimento esplicito traslazionale e circolare

#### 3DP6

- [ ] Testare movimento implicito e esplicito circolare

#### 3DP7

- [ ] Testare movimento implicito e esplicito traslazionale

#### 3DP8

- [ ] Testare movimento implicito e esplicito traslazionale e circolare

## Esperimenti su riconoscimento moto

Esperimenti che mirano a determinare se la rete riesca o meno a generalizzare il concetto di moto.

### Modello 2D e coerenza 60-90-120

#### 2DM0

- [ ] Allenare su implicito traslazionale -> Testare su implicito circolare

#### 2DM1

- [ ] Allenare su implicito circolare -> Testare su implicito traslazionale

#### 2DM2

- [ ] Allenare su implicito circolare e traslazionale -> Testare su implicito traslazionale

#### 2DM3

- [ ] Allenare su implicito circolare e traslazionale -> Testare su implicito circolare

### Modello 3D e coerenza 60-90-120

#### 3DM0

- [ ] Allenare su implicito traslazionale -> Testare su esplicito traslazionale

#### 3DM1

- [ ] Allenare su esplicito traslazionale -> Testare su implicito traslazionale

#### 3DM2

- [ ] Allenare su implicito circolare -> Testare su esplicito circolare

#### 3DM3

- [ ] Allenare su esplicito circolare -> Testare su implicito circolare

#### 3DM4

- [ ] Allenare su implicito traslazionale e circolare -> Testare su esplicito traslazionale e circolare

#### 3DM5

- [ ] Allenare su esplicito traslazionale e circolare -> Testare su implicito traslazionale e circolare

#### 3DM6

- [ ] Allenare su implicito e esplicito traslazionale -> Testare su implicito e esplicito circolare

#### 3DM7

- [ ] Allenare su implicito e esplicito circolare -> Testare su implicito e esplicito traslazionale

#### 3DM8

- [ ] Allenare su implicito e esplicito traslazionale e circolare -> Testare su implicito e esplicito traslazionale e circolare

## Esperimenti su specializzazione

Esperimenti che mirano ad evidenziare se la rete specializzi o meno alcuni strati nel riconoscimento di un determinato tipo di moto, per ogni modello (2D e 3D) e per ogni soglia di coerenza (60-90-120).

### Modelli 2D e coerenza 60-90-120

#### 2DS0

- [ ] Testare movimento implicito circolare rimuovendo uno strato alla volta

#### 2DS1

- [ ] Testare movimento implicito traslazionale rimuovendo uno strato alla volta

#### 2DS2

- [ ] Testare movimento implicito traslazionale e circolare rimuovendo uno strato alla volta

### Modelli 3D e coerenza 60-90-120

#### 3DS3

- [ ] Testare movimento esplicito circolare rimuovendo uno strato alla volta

#### 3DS4

- [ ] Testare movimento esplicito traslazionale rimuovendo uno strato alla volta

#### 3DS5

- [ ] Testare movimento esplicito traslazionale e circolare rimuovendo uno strato alla volta