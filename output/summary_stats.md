# Resumo Estatístico — Sionna RT Interlagos

**Frequência:** 3.5 GHz | **UEs simulados:** 1681 | **gNB:** 62 m AGL

**Estimador de posição:** d̂ = 10^((EIRP − RSRP) / (10·n)) com EIRP = -7 dBm (P_tx 23 dBm + ganho 8 dBi − PL₀ 38 dB), n = 2.7

| Categoria | n | % | RSRP med (dBm) | AoA err med (°) | AoA err p95 (°) | Pos err med (m) | Pos err p95 (m) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LoS | 887 | 52.8 | -63.1 | 0.0 | 0.0 | 88 | 409 |
| diffracted | 58 | 3.5 | -79.8 | 100.8 | 130.5 | 466 | 858 |
| reflected | 64 | 3.8 | -59.9 | 116.1 | 156.4 | 360 | 1486 |
| transmitted | 672 | 40.0 | -70.9 | 0.0 | 0.0 | 263 | 508 |
| **Total** | 1681 | 100.0 | -67.6 | 0.0 | 100.6 | 209 | 491 |

## Notas

- **LoS**: nenhuma interação no caminho dominante (linha de visada livre).
- **diffracted**: algum hop de difração no caminho dominante. AoA preservado, RSRP reduzido pela difração.
- **reflected**: algum hop especular/difuso (sem difração). AoA pode ser desviado para a superfície refletora.
- **transmitted**: somente hops de refração — travessia de parede (modo O2I). AoA preservado com boa precisão, RSRP atenuado ~17 dB vs LoS.
- **none**: nenhum caminho válido encontrado pelo PathSolver.
- AoA err = erro angular 3D entre AoA medido e direção geométrica real (produto interno de vetores unitários esféricos).
- Pos err = distância horizontal entre posição estimada e verdadeira.
- Bad regime (reflected + diffracted): UEs com maior probabilidade de erro AoA elevado, candidatos a outlier no estimador de posição.
