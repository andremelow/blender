% analyze_errors.m — Análise de erro de posição: 20k UEs espectadores Sionna RT
%
% Compara 3 estimadores RSRP × 4 categorias de caminho.
% Gera figura 4-painéis + resumo txt.
%
% Uso:
%   matlab -batch "addpath('MATLAB'); run('MATLAB/analyze_errors.m')"
%   (executar a partir do diretório raiz do projeto)
%
% Requer:
%   output/measurements_spectators_v2.mat
%   → gerar com: python scripts/export_to_matlab.py --source spectators

fprintf('=== analyze_errors.m | 20k UEs Sionna RT | Interlagos 3.5 GHz ===\n');

here = fileparts(mfilename('fullpath'));
if isempty(here); here = pwd; end
outDir = fullfile(here, '..', 'output');

matFile = fullfile(outDir, 'measurements_spectators_v2.mat');
if ~exist(matFile, 'file')
    error('analyze_errors:notFound', ...
        'Arquivo não encontrado: %s\nRode: python scripts/export_to_matlab.py --source spectators', ...
        matFile);
end

fprintf('Carregando: %s\n', matFile);
raw = load(matFile);

%% =========================================================
%  1. Extrai dados (1×N → colunas N×1)
%% =========================================================
rsrp   = raw.rsrp_dBm(:);
aoa_az = raw.aoa_az_rad(:);
aoa_el = raw.aoa_el_rad(:);
pos_e  = raw.pos_east(:);
pos_n  = raw.pos_north(:);
d_true = raw.d_true(:);
N      = double(raw.n_ues);

% path_type: cell(1,N) ou char array
if iscell(raw.path_type)
    pt_flat = raw.path_type(:);
else
    pt_flat = cellstr(raw.path_type(:));
end

fprintf('  UEs carregados: %d  |  RSRP válido: %d\n', ...
    int32(N), int32(sum(isfinite(rsrp))));

%% =========================================================
%  2. Parâmetros de calibração (embutidos no .mat pelo export)
%% =========================================================
n_naive  = raw.calib_n_naive;   A_naive  = raw.calib_A_naive;
n_global = raw.calib_n_global;  A_global = raw.calib_A_global;
n_los    = raw.calib_n_los;     A_los    = raw.calib_A_los;

D_MIN = 10;   D_MAX = 2000;

%% =========================================================
%  3. d_hat por estimador
%% =========================================================
dh_naive  = clamp_d(A_naive,  n_naive,  rsrp, D_MIN, D_MAX);
dh_global = clamp_d(A_global, n_global, rsrp, D_MIN, D_MAX);
dh_los    = clamp_d(A_los,    n_los,    rsrp, D_MIN, D_MAX);

%% =========================================================
%  4. Posição estimada 2D → erro (m)
%% =========================================================
% Conv. Sionna: aoa_az aponta gNB→UE (frame ENU).
% sin(pi/2 - aoa_el) = cos(aoa_el) projeta na horizontal.
proj_h = cos(aoa_el);

err_naive  = pos_err(dh_naive,  proj_h, aoa_az, pos_e, pos_n);
err_global = pos_err(dh_global, proj_h, aoa_az, pos_e, pos_n);
err_los    = pos_err(dh_los,    proj_h, aoa_az, pos_e, pos_n);

valid = isfinite(rsrp) & isfinite(err_naive);

%% =========================================================
%  5. Resumo no terminal
%% =========================================================
CATS   = {'LoS','reflected','diffracted','transmitted','none'};
EST_NAMES  = {'Naive (n=2.7)', 'Global calib.', 'LoS calib.'};
EST_ERR    = {err_naive, err_global, err_los};
EST_N      = {n_naive, n_global, n_los};
EST_A      = {A_naive, A_global, A_los};

fprintf('\n%s\n', repmat('=', 1, 72));
fprintf('%-15s %6s | %12s %12s %12s\n', 'Categoria', 'N', ...
    EST_NAMES{1}, EST_NAMES{2}, EST_NAMES{3});
fprintf('%s\n', repmat('-', 1, 72));
for ci = 1:numel(CATS)
    mask = strcmp(pt_flat, CATS{ci}) & valid;
    if ~any(mask); continue; end
    meds = cellfun(@(e) nanmedian(e(mask)), EST_ERR);
    fprintf('  %-13s %6d | %12.1f %12.1f %12.1f\n', ...
        CATS{ci}, sum(mask), meds(1), meds(2), meds(3));
end
fprintf('%s\n', repmat('-', 1, 72));
meds_all = cellfun(@(e) nanmedian(e(valid)), EST_ERR);
p95_all  = cellfun(@(e) prctile(e(valid), 95), EST_ERR);
fprintf('  %-13s %6d | %12.1f %12.1f %12.1f\n', 'TODOS', sum(valid), meds_all(1), meds_all(2), meds_all(3));
fprintf('  %-13s %6s | %12.1f %12.1f %12.1f\n', 'p95',  '', p95_all(1),  p95_all(2),  p95_all(3));
fprintf('%s\n', repmat('=', 1, 72));

%% =========================================================
%  6. Figura 4 painéis
%% =========================================================
PT_COLORS = [0.18 0.80 0.44;   % LoS       verde
             0.20 0.60 0.87;   % reflected azul
             0.90 0.49 0.13;   % diffracted laranja
             0.61 0.35 0.71;   % transmitted roxo
             0.55 0.55 0.55];  % none       cinza

EST_COLORS = [0.50 0.50 0.50;  % naive     cinza
              0.18 0.45 0.70;  % global    azul
              0.20 0.63 0.17]; % los       verde

fig = figure('Visible','off','Color','w','Position',[0 0 1600 1100]);

%% --- (a) Mapa scatter: UEs coloridos pelo erro [estimador LoS] ---------------
ax1 = subplot(2,2,1,'Parent',fig);
hold(ax1,'on');  grid(ax1,'on');  box(ax1,'on');
set(ax1,'Color',[0.97 0.97 0.97]);

% Subconjunto válido para scatter
idx_v = find(valid);
e_v   = pos_e(idx_v);   n_v   = pos_n(idx_v);
err_v = err_los(idx_v);

% Cor: log10(err+1) → colormap
log_err = log10(err_v + 1);
scatter(ax1, e_v, n_v, 4, log_err, 'filled', 'MarkerEdgeColor', 'none');
colormap(ax1, hot_r(256));
cb = colorbar(ax1);
cb.Label.String = 'log_{10}(err + 1) m';

% Ticks da colorbar em metros
err_ticks = [1 10 50 100 200 500 1000];
set(cb, 'Ticks', log10(err_ticks + 1), ...
    'TickLabels', arrayfun(@(x) sprintf('%d m',x), err_ticks, 'UniformOutput',false));

% Marcador gNB
plot(ax1, 0, 0, 'w^', 'MarkerSize', 12, 'MarkerFaceColor', 'r', ...
    'MarkerEdgeColor', 'k', 'LineWidth', 1.2, 'DisplayName', 'gNB');
text(ax1, 20, 20, 'gNB', 'Color','r','FontWeight','bold','FontSize',9);

% Legenda por path_type (símbolos)
for ci = 1:numel(CATS)
    mk = strcmp(pt_flat, CATS{ci});
    if ~any(mk); continue; end
    plot(ax1, NaN, NaN, 'o', 'Color', PT_COLORS(ci,:), ...
        'MarkerFaceColor', PT_COLORS(ci,:), 'MarkerSize', 5, ...
        'DisplayName', sprintf('%s (%d)', CATS{ci}, sum(mk)));
end
legend(ax1, 'show', 'Location', 'northeast', 'FontSize', 7);

title(ax1, sprintf('(a) Mapa de erro 2D — estimador LoS  (%d UEs)', sum(valid)), 'FontSize', 10);
xlabel(ax1, 'East relativo à gNB (m)');  ylabel(ax1, 'North relativo à gNB (m)');
axis(ax1, 'equal');

%% --- (b) CDF do erro por path_type [estimador LoS] --------------------------
ax2 = subplot(2,2,2,'Parent',fig);
hold(ax2,'on');  grid(ax2,'on');

for ci = 1:numel(CATS)
    mask = strcmp(pt_flat, CATS{ci}) & valid;
    if sum(mask) < 5; continue; end
    vals = sort(err_los(mask));
    cdf  = (1:numel(vals)) / numel(vals);
    plot(ax2, vals, cdf, 'Color', PT_COLORS(ci,:), 'LineWidth', 1.8, ...
        'DisplayName', sprintf('%s  (med=%.0f m, n=%d)', ...
        CATS{ci}, median(vals), numel(vals)));
end
% Todos
vals_all = sort(err_los(valid));
cdf_all  = (1:numel(vals_all)) / numel(vals_all);
plot(ax2, vals_all, cdf_all, 'k--', 'LineWidth', 1.5, ...
    'DisplayName', sprintf('Todos  (med=%.0f m)', median(vals_all)));

xline(ax2, prctile(vals_all, 50), ':', 'Color', [0.4 0.4 0.4], 'LineWidth', 1);
xline(ax2, prctile(vals_all, 95), '--', 'Color', [0.4 0.4 0.4], 'LineWidth', 1);
yline(ax2, 0.50, ':', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.8);
yline(ax2, 0.95, '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.8);

xlim(ax2, [0, 1500]);  ylim(ax2, [0, 1]);
legend(ax2, 'show', 'Location', 'lower right', 'FontSize', 8);
title(ax2, '(b) CDF erro por path\_type — estimador LoS calibrado', 'FontSize', 10);
xlabel(ax2, 'Erro de posição 2D (m)');  ylabel(ax2, 'CDF');

%% --- (c) Erro mediano vs distância real [3 estimadores] ---------------------
ax3 = subplot(2,2,3,'Parent',fig);
hold(ax3,'on');  grid(ax3,'on');

d_edges  = 0:100:1800;
d_mid    = d_edges(1:end-1) + 50;

for ei = 1:3
    err_ei = EST_ERR{ei};
    med_bin = nan(1, numel(d_mid));
    p25_bin = nan(1, numel(d_mid));
    p75_bin = nan(1, numel(d_mid));
    for bi = 1:numel(d_mid)
        mask_bin = valid & d_true >= d_edges(bi) & d_true < d_edges(bi+1);
        if sum(mask_bin) < 5; continue; end
        med_bin(bi) = nanmedian(err_ei(mask_bin));
        p25_bin(bi) = prctile(err_ei(mask_bin), 25);
        p75_bin(bi) = prctile(err_ei(mask_bin), 75);
    end
    plot(ax3, d_mid, med_bin, '-o', 'Color', EST_COLORS(ei,:), ...
        'MarkerSize', 4, 'LineWidth', 1.8, 'MarkerFaceColor', EST_COLORS(ei,:), ...
        'DisplayName', EST_NAMES{ei});
    % Banda p25-p75 apenas para LoS
    if ei == 3
        ok_b = ~isnan(med_bin);
        fill(ax3, [d_mid(ok_b), fliplr(d_mid(ok_b))], ...
            [p25_bin(ok_b), fliplr(p75_bin(ok_b))], ...
            EST_COLORS(ei,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none', ...
            'HandleVisibility', 'off');
    end
end

% Linha diagonal de referência (erro = distância → estimativa = gNB)
plot(ax3, d_mid, d_mid, 'k:', 'LineWidth', 0.8, 'DisplayName', 'ref: err=d');

% n por bin (eixo secundário)
ax3b = axes('Position', get(ax3,'Position'), 'YAxisLocation','right', ...
    'Color','none', 'XColor','none');
n_bin = arrayfun(@(bi) sum(valid & d_true>=d_edges(bi) & d_true<d_edges(bi+1)), ...
    1:numel(d_mid));
bar(ax3b, d_mid, n_bin, 100, 'FaceColor',[0.85 0.85 0.85], 'EdgeColor','none', ...
    'FaceAlpha', 0.4);
ylabel(ax3b, 'N por bin', 'FontSize', 8, 'Color', [0.5 0.5 0.5]);
set(ax3b,'YColor',[0.5 0.5 0.5],'XLim',[0 1800]);
uistack(ax3b, 'bottom');
axes(ax3);

xlim(ax3, [0, 1800]);  ylim(ax3, [0, 800]);
legend(ax3, 'show', 'Location', 'upper left', 'FontSize', 8);
title(ax3, '(c) Mediana do erro vs distância real (bins 100 m)', 'FontSize', 10);
xlabel(ax3, 'd_{true} (m)');  ylabel(ax3, 'Mediana do erro 2D (m)');

%% --- (d) Mediana por path_type × estimador (barras agrupadas) ---------------
ax4 = subplot(2,2,4,'Parent',fig);
hold(ax4,'on');  grid(ax4,'on');

CATS_PLOT = {'LoS','reflected','diffracted','transmitted'};
Nc = numel(CATS_PLOT);
x  = 1:Nc;
w  = 0.22;

for ei = 1:3
    err_ei = EST_ERR{ei};
    meds_ci = nan(1, Nc);
    for ci = 1:Nc
        mask = strcmp(pt_flat, CATS_PLOT{ci}) & valid;
        if any(mask)
            meds_ci(ci) = nanmedian(err_ei(mask));
        end
    end
    xpos = x + (ei - 2) * w;
    bar(ax4, xpos, meds_ci, w, ...
        'FaceColor', EST_COLORS(ei,:), 'EdgeColor', 'k', 'LineWidth', 0.5, ...
        'FaceAlpha', 0.85, 'DisplayName', EST_NAMES{ei});
    % Anota valor sobre barra
    for ci = 1:Nc
        if ~isnan(meds_ci(ci))
            text(ax4, xpos(ci), meds_ci(ci) + 5, sprintf('%.0f', meds_ci(ci)), ...
                'HorizontalAlignment','center','FontSize',6.5,'Color','k');
        end
    end
end

% n por categoria sob os eixos
for ci = 1:Nc
    n_ci = sum(strcmp(pt_flat, CATS_PLOT{ci}) & valid);
    text(ax4, x(ci), -30, sprintf('n=%d', n_ci), ...
        'HorizontalAlignment','center','FontSize',7,'Color',[0.4 0.4 0.4]);
end

set(ax4, 'XTick', x, 'XTickLabel', CATS_PLOT, 'XTickLabelRotation', 0);
% Escala automática: máximo da mediana entre estimadores e categorias
med_max = 0;
for ei_s = 1:3
    for ci_s = 1:Nc
        mk_s = strcmp(pt_flat, CATS_PLOT{ci_s}) & valid;
        if any(mk_s)
            med_max = max(med_max, nanmedian(EST_ERR{ei_s}(mk_s)));
        end
    end
end
ylim(ax4, [-50, med_max * 1.30]);
legend(ax4, 'show', 'Location', 'northeast', 'FontSize', 8);
title(ax4, '(d) Mediana do erro por path\_type e estimador', 'FontSize', 10);
xlabel(ax4, 'Categoria');  ylabel(ax4, 'Mediana do erro 2D (m)');

%% Título global
sgtitle(fig, sprintf( ...
    'Análise de Erro de Posição — Interlagos 3.5 GHz  |  %d UEs espectadores  |  gNB h=62 m', ...
    sum(valid)), 'FontSize', 12, 'FontWeight', 'bold');

%% =========================================================
%  7. Salva figura
%% =========================================================
outPng = fullfile(outDir, 'position_errors_20k.png');
print(fig, outPng, '-dpng', '-r150');
fprintf('\nFigura salva: %s\n', outPng);
close(fig);

%% =========================================================
%  8. Arquivo txt com métricas detalhadas
%% =========================================================
outTxt = fullfile(outDir, 'position_errors_20k.txt');
fid = fopen(outTxt, 'w');
fprintf(fid, 'Análise de Erro de Posição — Interlagos 3.5 GHz\n');
fprintf(fid, 'Data: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, 'UEs analisados: %d / %d  (RSRP inválido: %d)\n\n', ...
    sum(valid), int32(N), int32(N) - sum(valid));

fprintf(fid, 'Estimadores:\n');
for ei = 1:3
    fprintf(fid, '  %d. %s  (n=%.3f, A=%.2f dBm)\n', ...
        ei, EST_NAMES{ei}, EST_N{ei}, EST_A{ei});
end
fprintf(fid, '\n');

fprintf(fid, '%-15s %6s | %14s %14s %14s\n', ...
    'Categoria', 'N', EST_NAMES{1}, EST_NAMES{2}, EST_NAMES{3});
fprintf(fid, '%s\n', repmat('-', 1, 78));

for ci = 1:numel(CATS)
    mask = strcmp(pt_flat, CATS{ci}) & valid;
    if ~any(mask); continue; end
    e1 = err_naive(mask);  e2 = err_global(mask);  e3 = err_los(mask);
    fprintf(fid, '  %-13s %6d | %8.1f (p95:%5.0f) %8.1f (p95:%5.0f) %8.1f (p95:%5.0f)\n', ...
        CATS{ci}, sum(mask), ...
        nanmedian(e1), prctile(e1,95), ...
        nanmedian(e2), prctile(e2,95), ...
        nanmedian(e3), prctile(e3,95));
end

fprintf(fid, '%s\n', repmat('-', 1, 78));
e1 = err_naive(valid);  e2 = err_global(valid);  e3 = err_los(valid);
fprintf(fid, '  %-13s %6d | %8.1f (p95:%5.0f) %8.1f (p95:%5.0f) %8.1f (p95:%5.0f)\n', ...
    'TODOS', sum(valid), ...
    nanmedian(e1), prctile(e1,95), ...
    nanmedian(e2), prctile(e2,95), ...
    nanmedian(e3), prctile(e3,95));
fprintf(fid, '\n');

fprintf(fid, 'Percentis do erro 2D — todos os UEs:\n');
pcts = [10 25 50 75 90 95 99];
for ei = 1:3
    ev = EST_ERR{ei};
    fprintf(fid, '  %s:\n', EST_NAMES{ei});
    for pp = pcts
        fprintf(fid, '    p%02d = %6.1f m\n', pp, prctile(ev(valid), pp));
    end
end

fprintf(fid, '\nBias de distância (mediana d_hat / d_true por categoria):\n');
fprintf(fid, '%-15s | %10s %10s %10s\n', 'Categoria', EST_NAMES{1}, EST_NAMES{2}, EST_NAMES{3});
fprintf(fid, '%s\n', repmat('-', 1, 50));
for ci = 1:numel(CATS)
    mask = strcmp(pt_flat, CATS{ci}) & valid;
    if ~any(mask); continue; end
    dt = d_true(mask);
    dhi_mat = [dh_naive(mask), dh_global(mask), dh_los(mask)];
    bias = nan(1, 3);
    for ei = 1:3
        bias(ei) = nanmedian(dhi_mat(:, ei) ./ dt);
    end
    fprintf(fid, '  %-13s | %9.2fx %10.2fx %10.2fx\n', CATS{ci}, bias(1), bias(2), bias(3));
end

fclose(fid);
fprintf('Métricas salvas: %s\n', outTxt);
fprintf('=== Concluído. ===\n');

%% =========================================================
%  Funções locais
%% =========================================================
function dh = clamp_d(A, n, rsrp_vec, d_min, d_max)
%CLAMP_D  d_hat = 10^((A - RSRP) / (10n)), limitado a [d_min, d_max].
dh = 10.^((A - rsrp_vec) ./ (10 * n));
dh = min(max(dh, d_min), d_max);
end

function err = pos_err(dh, proj_h, aoa_az, pos_e, pos_n)
%POS_ERR  Erro 2D (m) entre posição estimada e verdadeira.
%  proj_h = cos(aoa_el); aoa_az aponta gNB→UE (conv. Sionna).
xe  = dh .* proj_h .* cos(aoa_az);
ye  = dh .* proj_h .* sin(aoa_az);
err = sqrt((xe - pos_e).^2 + (ye - pos_n).^2);
end

function cmap = hot_r(N)
%HOT_R  Colormap hot reverso: branco → amarelo → laranja → vermelho → preto.
if nargin < 1; N = 256; end
cmap = flipud(hot(N));
end
