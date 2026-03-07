# teclahacker ⌨️🎧

Uma implementação prática de um ataque de canal lateral acústico em teclados (Acoustic Side-Channel Attack) utilizando Deep Learning. 

Este projeto é inspirado no artigo científico localizado em `etc/` (*A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards*) e demonstra como o som das teclas pode ser usado para inferir o que está sendo digitado utilizando Redes Neurais Convolucionais (CNNs) e Espectrogramas de Mel.

> *point is, reimplement the paper on your own terms.... hackee as emanacoes do teclador*

⚠️ **Aviso:** Este projeto tem fins estritamente **educacionais e de pesquisa**. Não utilize para fins maliciosos ou para interceptar dados sem autorização.

## Requisitos

- Python >= 3.11, < 3.13
- [Poetry](https://python-poetry.org/) para gerenciamento de dependências.

## Instalação

Na raiz do projeto, instale as dependências executando:

```bash
poetry install
```

Isso instalará as bibliotecas necessárias, incluindo `torch`, `torchaudio`, `librosa`, `numpy` e `scipy`.

## Experimento Rápido (Dummy Data)

Para testar a pipeline de ponta a ponta sem precisar gravar horas de áudio do seu próprio teclado, incluímos um gerador de dados sintéticos.

**1. Gerar o dataset de teste e o áudio alvo:**
```bash
poetry run python gerar_exemplo_zorro.py
```
Isso criará uma pasta `dummy_data/` com áudios curtos simulando as teclas necessárias e um arquivo `o_zorro_e_gris.wav` simulando alguém digitando a frase "o zorro e gris".

**2. Treinar a Rede Neural:**
```bash
poetry run python teclahacker.py --train dummy_data/
```
O script processará os áudios, extrairá os Mel-Spectrograms e treinará a CNN. No final, salvará o modelo (`keystroke_model.pth`) e as classes reconhecidas (`classes.txt`).

**3. Testar a "escuta" (Predição):**
```bash
poetry run python teclahacker.py --predict o_zorro_e_gris.wav
```
O modelo tentará decodificar o áudio simulado e deverá imprimir algo muito próximo de `o zorro e gris`.

## Usando com um Teclado Real

Para usar com o seu próprio teclado, siga esta estrutura:

1. **Gravação e Organização:** Grave você digitando repetidamente cada tecla separadamente. Organize os arquivos `.wav` em pastas com o nome da respectiva tecla:
   ```text
   meu_teclado_data/
   ├── a/
   │   ├── 1.wav
   │   ├── 2.wav
   │   └── ...
   ├── b/
   ├── space/
   └── ...
   ```
2. **Treinamento:** `poetry run python teclahacker.py --train meu_teclado_data/`
3. **Ataque:** Grave um áudio contínuo digitando uma senha ou frase e use `poetry run python teclahacker.py --predict sua_gravacao.wav`.

## Como funciona?

1. **Extração de Onset:** O `librosa` é usado para detectar os picos de energia no áudio (o exato momento do clique da tecla).
2. **Mel-Spectrograms:** O trecho de áudio de cada clique é convertido em uma representação visual (espectrograma de Mel), que captura as frequências ao longo do tempo.
3. **CNN:** Uma Rede Neural Convolucional (implementada em PyTorch) recebe essa imagem do som e classifica de qual tecla ela pertence.
