# Decomposição do Erro de Posição

**Frequência:** 3.5 GHz | **Estimador global:** n=1.362, A=-32.95 dBm

**err_tangencial** = d_true · sin(min(|ε|, π − |ε|)) — ângulo efetivo entre vetor estimado e eixo gNB→UE

| Categoria | n | Radial med (m) | Tang. med (m) | Dominante |
|:---|---:|---:|---:|:---:|
| LoS | 887 | 366.8 | 0.0 | **radial** |
| diffracted | 58 | 2465.5 | 175.0 | **radial** |
| reflected | 64 | 273.4 | 227.3 | **radial** |
| transmitted | 672 | 347.4 | 0.0 | **radial** |

## Verificação

err_total_predito = √(radial² + tang²) deve aproximar o erro 2D real.
Discrepância esperada: projeção 3D→2D e linearização d_hat ≈ d_true.
