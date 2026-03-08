# teclahacker ⌨️🎧

Uma implementação prática de um ataque de canal lateral acústico em teclados (Acoustic Side-Channel Attack) utilizando Deep Learning. 

Demonstra como o som das teclas pode ser usado para inferir o que está sendo digitado utilizando Redes Neurais Convolucionais (CNNs) e Espectrogramas de Mel.

> *ponto é, implementar em seus próprios termos.... hackee as emanações do teclador*

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

## Experimento Rápido (Dados Falsos)

Para testar o fluxo de ponta a ponta sem precisar gravar horas de áudio do seu próprio teclado, incluímos um gerador de dados sintéticos.

**1. Gerar o conjunto de dados de teste e o áudio alvo:**
```bash
poetry run python gerar_exemplo_zorro.py
```
Isso criará uma pasta `dados_falsos/` com áudios curtos simulando as teclas necessárias e um arquivo `o_zorro_e_gris.wav` simulando alguém digitando a frase "o zorro e gris".

**2. Treinar a Rede Neural:**
```bash
poetry run python teclahacker.py --treinar dados_falsos/
```
O script processará os áudios, extrairá os Espectrogramas de Mel e treinará a CNN. No final, salvará o modelo (`modelo_teclado.pth`) e as classes reconhecidas (`classes.txt`).

**3. Testar a "escuta" (Predição):**
```bash
poetry run python teclahacker.py --prever o_zorro_e_gris.wav
```
O modelo tentará decodificar o áudio simulado e deverá imprimir algo muito próximo de `o zorro e gris`.

## Usando com um Teclado Real

Para usar com o seu próprio teclado, siga esta estrutura:

1. **Gravação e Organização:** Grave você digitando repetidamente cada tecla separadamente. Organize os arquivos `.wav` em pastas com o nome da respectiva tecla:
   ```text
   dados_meu_teclado/
   ├── a/
   │   ├── 1.wav
   │   ├── 2.wav
   │   └── ...
   ├── b/
   ├── espaco/
   └── ...
   ```
2. **Treinamento:** `poetry run python teclahacker.py --treinar dados_meu_teclado/`
3. **Ataque:** Grave um áudio contínuo digitando uma senha ou frase e use `poetry run python teclahacker.py --prever sua_gravacao.wav`.

## Como funciona?

1. **Extração de Início (Onset):** A biblioteca `librosa` é usada para detectar os picos de energia no áudio (o exato momento do clique da tecla).
2. **Espectrogramas de Mel:** O trecho de áudio de cada clique é convertido em uma representação visual (espectrograma de Mel), que captura as frequências ao longo do tempo.
3. **CNN:** Uma Rede Neural Convolucional (implementada em PyTorch) recebe essa imagem do som e classifica de qual tecla ela pertence.

## Referências

```bibtex
@misc{henry2026quantummnist,
  author = {Julian Henry},
  title = {quantum-mnist},
  year = {2026},
  organization = {Aeae.inc},
  address = {Houston, Texas},
  note = {Software repository}
}

@misc{harrison2023practical,
  title={A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards}, 
  author={Joshua Harrison and Ehsan Toreini and Maryam Mehrnezhad},
  year={2023},
  eprint={2308.01074},
  archivePrefix={arXiv},
  primaryClass={cs.CR}
}

@inproceedings{zhuang2005keyboard,
  title={Keyboard acoustic emanations revisited},
  author={Zhuang, Li and Zhou, Feng and Tygar, J. D.},
  booktitle={Proceedings of the 12th ACM conference on Computer and communications security},
  pages={373--382},
  year={2005}
}
```
