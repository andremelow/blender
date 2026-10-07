# Estimador ToA vs RSRP — 3.5 GHz

**c × τ_0**: velocidade da luz × delay de propagação absoluto
**τ_dom**: caminho dominante (maior potência)
**τ_min**: primeira chegada (menor delay)

| Estimador | Todos med (m) | Todos p95 (m) |
|:---|---:|---:|
| RSRP naive n=2.7 | 208.5 | 490.6 |
| RSRP global calib. | 376.5 | 7570.2 |
| ToA dom. path | 0.0 | 199.9 |
| ToA 1ª chegada | 0.0 | 168.4 |

## Por categoria — mediana do erro de posição (m)

| Categoria | n | RSRP naive | RSRP calib | ToA dom | **ToA min** |
|:---|---:|---:|---:|---:|---:|
| LoS | 887 | 88.4 | 361.4 | 0.0 | 0.0 |
| diffracted | 58 | 466.5 | 2472.3 | 292.5 | 277.3 |
| reflected | 64 | 360.5 | 425.4 | 457.2 | 431.9 |
| transmitted | 672 | 262.9 | 336.8 | 0.0 | 0.0 |

## Achado central

τ_min (primeira chegada) corresponde ao caminho LoS em quase todos os UEs — mesmo naqueles classificados como 'reflected' ou 'diffracted', a componente LoS existe e chega primeiro, apenas com menor potência que o caminho refletido/difratado. Resultado: `d_hat_toa = c × τ_min ≈ d_true` para todos os tipos de caminho. O erro de posição residual é dominado pelo erro de AoA (direção do caminho dominante).
