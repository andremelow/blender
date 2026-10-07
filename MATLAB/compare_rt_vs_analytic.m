% compare_rt_vs_analytic.m — Comparação RT vs Analítico para o paper.
%
% Gera output/rho_comparison.png com 4 painéis e
%         output/rho_comparison_metrics.txt com métricas numéricas.
%
% Uso: matlab -batch "addpath('MATLAB'); run('MATLAB/compare_rt_vs_analytic.m')"
%      (executar a partir do diretório raiz do projeto)
%
% Mapas calculados no mesmo grid do demo (cellSize=50m):
%   rho_true     — UEs verdadeiros convolvidos com gaussiana sigma=20m
%   rho_analytic — 20x simulate_measurements (MC) + local_gnb_build_density_map
%   rho_rt       — 1x load_measurements (Sionna RT) + local_gnb_build_density_map
%
% Restrições: simulate_measurements e local_gnb_build_density_map não
% são modificados (cópias literais do original abaixo).

fprintf('=== compare_rt_vs_analytic.m ===\n');

here = fileparts(mfilename('fullpath'));
if isempty(here); here = pwd; end
addpath(here);   % garante load_measurements.m no path

outDir = fullfile(here, '..', 'output');

%% =========================================================
%  Parâmetros — idênticos a gnb_density_demo_geo.m
%% =========================================================
centerLat = -(23 + 42/60 + 06.9/3600);
centerLon = -(46 + 42/60 + 03.4/3600);

halfEast  = 1600;   halfNorth = 600;
mapXlim   = [-halfEast,  +halfEast];
mapYlim   = [-halfNorth, +halfNorth];
zoomLvl   = 16;
cellSize  = 50;

xVec = (mapXlim(1)+cellSize/2):cellSize:(mapXlim(2)-cellSize/2);
yVec = (mapYlim(1)+cellSize/2):cellSize:(mapYlim(2)-cellSize/2);
[XG, YG] = meshgrid(xVec, yVec);
Nx = numel(xVec);   Ny = numel(yVec);

gnb = struct('pos',[0, 0, -62], 'cellId',1);

% Parâmetros do estimador (default do demo — não tunar)
params = struct( ...
    'n',             2.7,         ...
    'PL0_dB',        38,          ...
    'Pt_dBm',        23,          ...
    'Gt_dB',         8,           ...
    'Gr_dB',         0,           ...
    'sigma_xi_dB',   6,           ...
    'sigma_phi_rad', deg2rad(3),  ...
    'd_min',         10,          ...
    'd_max',         1800,        ...
    'cap_sigma_m',   250,         ...
    'rho_op',        5e-4 );

N_MC = 20;   % iterações Monte Carlo para rho_analytic
SIGMA_TRUE = 20;   % gaussiana do ground-truth (m)

%% =========================================================
%  1. Carrega posições verdadeiras dos UEs Sionna RT
%% =========================================================
matFile = fullfile(outDir, 'measurements_rt.mat');
fprintf('Carregando posições verdadeiras: %s\n', matFile);
raw = load(matFile);
ueTrue = [raw.pos_east, raw.pos_north];   % (N×2) relativo à gNB
N = size(ueTrue, 1);
fprintf('  %d UEs carregados.\n', N);

%% =========================================================
%  2. rho_true: massa pontual + gaussiana sigma=20m
%% =========================================================
fprintf('Calculando rho_true (sigma=%d m)...\n', SIGMA_TRUE);
t0 = tic;
rho_true_raw = zeros(Ny, Nx);
sig2 = SIGMA_TRUE^2;
Rtrunc = 4 * SIGMA_TRUE;

for k = 1:N
    xk = ueTrue(k,1);  yk = ueTrue(k,2);
    ix = find(xVec >= xk - Rtrunc & xVec <= xk + Rtrunc);
    iy = find(yVec >= yk - Rtrunc & yVec <= yk + Rtrunc);
    if isempty(ix) || isempty(iy); continue; end
    Xs = XG(iy, ix);   Ys = YG(iy, ix);
    g  = exp(-0.5*((Xs-xk).^2 + (Ys-yk).^2) / sig2) / (2*pi*sig2);
    rho_true_raw(iy, ix) = rho_true_raw(iy, ix) + g;
end
rho_true = min(rho_true_raw / params.rho_op, 1);
t_true = toc(t0);
fprintf('  rho_true em %.2f s | max=%.4f | massa=%.1f\n', ...
    t_true, max(rho_true(:)), ...
    sum(rho_true(:)) * params.rho_op * cellSize^2);

%% =========================================================
%  3. rho_analytic: Monte Carlo 20x com simulate_measurements
%% =========================================================
fprintf('Calculando rho_analytic (%d iterações MC)...\n', N_MC);
t0 = tic;
rho_analytic_acc = zeros(Ny, Nx);

for mc = 1:N_MC
    if mod(mc, 5) == 0
        fprintf('  MC %d/%d...\n', mc, N_MC);
    end
    ueRaw_mc = simulate_measurements(ueTrue, gnb, params);
    rhoMap_mc = local_gnb_build_density_map(gnb, ueRaw_mc, XG, YG, params);
    rho_analytic_acc = rho_analytic_acc + rhoMap_mc;
end
rho_analytic = rho_analytic_acc / N_MC;
t_analytic = toc(t0);
fprintf('  rho_analytic em %.2f s | max=%.4f | massa=%.1f\n', ...
    t_analytic, max(rho_analytic(:)), ...
    sum(rho_analytic(:)) * params.rho_op * cellSize^2);

%% =========================================================
%  4. rho_rt: single shot com load_measurements (Sionna RT)
%% =========================================================
fprintf('Calculando rho_rt (Sionna RT)...\n');
t0 = tic;
ueRaw_rt = load_measurements(ueTrue, gnb, params);
rho_rt   = local_gnb_build_density_map(gnb, ueRaw_rt, XG, YG, params);
t_rt = toc(t0);
fprintf('  rho_rt em %.2f s | max=%.4f | massa=%.1f\n', ...
    t_rt, max(rho_rt(:)), ...
    sum(rho_rt(:)) * params.rho_op * cellSize^2);

%% =========================================================
%  5. Métricas
%% =========================================================
% MAE por célula
MAE_analytic = mean(abs(rho_analytic(:) - rho_true(:)));
MAE_rt       = mean(abs(rho_rt(:)       - rho_true(:)));

% Bias de massa total
mass_true     = sum(rho_true(:))     * params.rho_op * cellSize^2;
mass_analytic = sum(rho_analytic(:)) * params.rho_op * cellSize^2;
mass_rt       = sum(rho_rt(:))       * params.rho_op * cellSize^2;

% Hotspot recovery: top-3 hotspots em rho_true (separados por >= 100m)
topHotspots = [];    % (k×2) [East, North]
rtSorted = sortrows([rho_true(:), (1:Ny*Nx)'], -1);
for si = 1:size(rtSorted, 1)
    idx = rtSorted(si, 2);
    [iy, ix] = ind2sub([Ny Nx], idx);
    xc = xVec(ix);  yc = yVec(iy);
    if isempty(topHotspots) || ...
       all(sqrt((topHotspots(:,1)-xc).^2 + (topHotspots(:,2)-yc).^2) > 100)
        topHotspots(end+1, :) = [xc, yc]; %#ok<AGROW>
    end
    if size(topHotspots, 1) >= 3; break; end
end

% Para cada hotspot verdadeiro, distância ao pico mais próximo em cada mapa
% "Pico" = célula com valor local máximo (>= vizinhos 4-conexos)
is_peak_analytic = false(Ny, Nx);
is_peak_rt       = false(Ny, Nx);
for iy = 2:Ny-1
    for ix = 2:Nx-1
        if rho_analytic(iy,ix) >= rho_analytic(iy-1,ix) && ...
           rho_analytic(iy,ix) >= rho_analytic(iy+1,ix) && ...
           rho_analytic(iy,ix) >= rho_analytic(iy,ix-1) && ...
           rho_analytic(iy,ix) >= rho_analytic(iy,ix+1) && ...
           rho_analytic(iy,ix) > 0.05
            is_peak_analytic(iy,ix) = true;
        end
        if rho_rt(iy,ix) >= rho_rt(iy-1,ix) && ...
           rho_rt(iy,ix) >= rho_rt(iy+1,ix) && ...
           rho_rt(iy,ix) >= rho_rt(iy,ix-1) && ...
           rho_rt(iy,ix) >= rho_rt(iy,ix+1) && ...
           rho_rt(iy,ix) > 0.05
            is_peak_rt(iy,ix) = true;
        end
    end
end

[peak_an_y, peak_an_x] = find(is_peak_analytic);
[peak_rt_y, peak_rt_x] = find(is_peak_rt);
peaks_an = [xVec(peak_an_x)', yVec(peak_an_y)'];
peaks_rt = [xVec(peak_rt_x)', yVec(peak_rt_y)'];

dist_hotspot_an = nan(3,1);
dist_hotspot_rt = nan(3,1);

for h = 1:size(topHotspots,1)
    xh = topHotspots(h,1);  yh = topHotspots(h,2);
    if ~isempty(peaks_an)
        d_an = sqrt((peaks_an(:,1)-xh).^2 + (peaks_an(:,2)-yh).^2);
        dist_hotspot_an(h) = min(d_an);
    end
    if ~isempty(peaks_rt)
        d_rt = sqrt((peaks_rt(:,1)-xh).^2 + (peaks_rt(:,2)-yh).^2);
        dist_hotspot_rt(h) = min(d_rt);
    end
end

%% Resumo no terminal
fprintf('\n%s\n', repmat('=',1,60));
fprintf('%-30s %10s %10s\n', 'Métrica', 'Analítico', 'RT');
fprintf('%s\n', repmat('-',1,60));
fprintf('%-30s %10.4f %10.4f\n', 'MAE por célula',         MAE_analytic, MAE_rt);
fprintf('%-30s %10.1f %10.1f\n', 'Massa estimada (target)', mass_analytic, mass_rt);
fprintf('%-30s %10.1f\n',        'Massa ground-truth',      mass_true);
for h = 1:3
    fprintf('%-30s %10.1f %10.1f\n', ...
        sprintf('Hotspot %d dist. (m)', h), dist_hotspot_an(h), dist_hotspot_rt(h));
end
fprintf('%s\n', repmat('=',1,60));

%% =========================================================
%  6. Basemap OSM
%% =========================================================
fprintf('Baixando tiles OSM...\n');
try
    [bgRGB, ~, ~] = osm_basemap(centerLat, centerLon, halfNorth, halfEast, zoomLvl);
    hasBasemap = true;
    fprintf('  OSM OK.\n');
catch ME
    fprintf('  Aviso: OSM indisponível (%s) — fundo branco.\n', ME.message);
    hasBasemap = false;
    bgRGB = ones(Ny, Nx, 3);
end

%% =========================================================
%  7. Figura 4 painéis
%% =========================================================
fig = figure('Visible','off','Color','w','Position',[0 0 1600 900]);

% Scale comum para painéis a, b, c
clim_max = max([max(rho_true(:)), max(rho_analytic(:)), max(rho_rt(:))]);
clim_max = max(clim_max, 0.01);   % evita escala zerada

% Painel d: escala simétrica no erro diferencial
delta_err = abs(rho_rt - rho_true) - abs(rho_analytic - rho_true);
clim_diff = max(abs(delta_err(:)));
clim_diff = max(clim_diff, 1e-4);

alpha_img = 0.70;   % opacidade do heatmap sobre basemap

helper_draw_bg = @(ax) draw_basemap_on(ax, bgRGB, mapXlim, mapYlim, hasBasemap);

ax = gobjects(1,4);
for pi = 1:4
    ax(pi) = subplot(2,2,pi,'Parent',fig);
end

%% (a) rho_true com pontos UE
helper_draw_bg(ax(1));
hold(ax(1),'on');
hIa = imagesc(ax(1), xVec, yVec, rho_true);
set(hIa,'AlphaData', alpha_img * (rho_true / clim_max));
caxis(ax(1), [0 clim_max]);
colormap(ax(1), red_ramp(256));
scatter(ax(1), ueTrue(:,1), ueTrue(:,2), 4, [0.1 0.3 0.9], ...
        'filled','MarkerFaceAlpha',0.4,'MarkerEdgeColor','none');
plot(ax(1), 0, 0, 'r^','MarkerSize',12,'MarkerFaceColor','r','LineWidth',1.2);
title(ax(1), sprintf('(a) Ground truth  (\\sigma_{gauss}=%d m)', SIGMA_TRUE), ...
      'FontSize',10);
xlim(ax(1),mapXlim);  ylim(ax(1),mapYlim);
xlabel(ax(1),'East (m)');  ylabel(ax(1),'North (m)');
colorbar(ax(1));

%% (b) rho_analytic
helper_draw_bg(ax(2));
hold(ax(2),'on');
hIb = imagesc(ax(2), xVec, yVec, rho_analytic);
set(hIb,'AlphaData', alpha_img * (rho_analytic / clim_max));
caxis(ax(2), [0 clim_max]);
colormap(ax(2), red_ramp(256));
plot(ax(2), 0, 0, '^','MarkerSize',12,'MarkerFaceColor','w', ...
     'MarkerEdgeColor','k','LineWidth',1.2);
title(ax(2), sprintf('(b) Analítico  (MC=%d)  |  MAE=%.4f  |  massa=%.0f', ...
      N_MC, MAE_analytic, mass_analytic), 'FontSize',10);
xlim(ax(2),mapXlim);  ylim(ax(2),mapYlim);
xlabel(ax(2),'East (m)');  ylabel(ax(2),'North (m)');
colorbar(ax(2));

%% (c) rho_rt
helper_draw_bg(ax(3));
hold(ax(3),'on');
hIc = imagesc(ax(3), xVec, yVec, rho_rt);
set(hIc,'AlphaData', alpha_img * (rho_rt / clim_max));
caxis(ax(3), [0 clim_max]);
colormap(ax(3), red_ramp(256));
plot(ax(3), 0, 0, '^','MarkerSize',12,'MarkerFaceColor','w', ...
     'MarkerEdgeColor','k','LineWidth',1.2);
title(ax(3), sprintf('(c) Sionna RT  (det.)  |  MAE=%.4f  |  massa=%.0f', ...
      MAE_rt, mass_rt), 'FontSize',10);
xlim(ax(3),mapXlim);  ylim(ax(3),mapYlim);
xlabel(ax(3),'East (m)');  ylabel(ax(3),'North (m)');
colorbar(ax(3));

%% (d) Erro diferencial |rho_rt - true| - |rho_analytic - true|
helper_draw_bg(ax(4));
hold(ax(4),'on');
hId = imagesc(ax(4), xVec, yVec, delta_err);
set(hId,'AlphaData', alpha_img * abs(delta_err) / max(clim_diff, 1e-9));
caxis(ax(4), [-clim_diff  clim_diff]);
colormap(ax(4), diverging_bwr(256));
cb4 = colorbar(ax(4));
cb4.Label.String = '|err_{RT}| − |err_{analytic}|';
plot(ax(4), 0, 0, '^','MarkerSize',12,'MarkerFaceColor','k', ...
     'MarkerEdgeColor','w','LineWidth',1.2);

% Hotspots verdadeiros (cruzes pretas)
for h = 1:size(topHotspots,1)
    plot(ax(4), topHotspots(h,1), topHotspots(h,2), 'k+', ...
         'MarkerSize',14,'LineWidth',2.5);
end

title(ax(4), ['(d) Erro diferencial  (azul=RT melhor, vermelho=RT pior)'...
              sprintf('\n+ = hotspot verdadeiro')], 'FontSize',9);
xlim(ax(4),mapXlim);  ylim(ax(4),mapYlim);
xlabel(ax(4),'East (m)');  ylabel(ax(4),'North (m)');

%% Título global
sgtitle(fig, sprintf( ...
    'Comparação RT vs Analítico — Interlagos 3.5 GHz | %d UEs | n=%.1f, EIRP=%d dBm\\newline', ...
    N, params.n, round(params.Pt_dBm + params.Gt_dB - params.PL0_dB)), ...
    'FontSize',11,'FontWeight','bold');

%% Salva figura
outPng = fullfile(outDir, 'rho_comparison.png');
print(fig, outPng, '-dpng', '-r150');
fprintf('Figura salva: %s\n', outPng);
close(fig);

%% =========================================================
%  8. Métricas detalhadas por path_type (arquivo txt)
%% =========================================================
% Determina path_type dominante por célula com base nas posições dos UEs
path_types_ue = raw.path_type;   % cell(N,1)
cat_list = {'LoS','reflected','diffracted','transmitted','none'};

% Mapeia cada célula da grade ao path_type do UE mais próximo
cell_centers_E = XG(:);   cell_centers_N = YG(:);
dist_ue = zeros(numel(cell_centers_E), N);
for k = 1:N
    dist_ue(:,k) = sqrt((cell_centers_E - ueTrue(k,1)).^2 + ...
                        (cell_centers_N - ueTrue(k,2)).^2);
end
[~, nearest_ue] = min(dist_ue, [], 2);
cell_path_type = path_types_ue(nearest_ue);   % cell(Ny*Nx, 1)

outTxt = fullfile(outDir, 'rho_comparison_metrics.txt');
fid = fopen(outTxt, 'w');
fprintf(fid, 'Comparação RT vs Analítico — Interlagos 3.5 GHz\n');
fprintf(fid, 'Data: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, 'UEs: %d | Grid: %dx%d (cellSize=%d m)\n\n', ...
    N, Ny, Nx, cellSize);

fprintf(fid, '%-20s %10s %10s %10s\n', 'Categoria', 'n_celulas', 'MAE_an', 'MAE_rt');
fprintf(fid, '%s\n', repmat('-',1,55));
for ci = 1:numel(cat_list)
    cat = cat_list{ci};
    mask = strcmp(cell_path_type, cat);
    if ~any(mask); continue; end
    mae_an = mean(abs(rho_analytic(mask) - rho_true(mask)));
    mae_rt = mean(abs(rho_rt(mask)       - rho_true(mask)));
    fprintf(fid, '%-20s %10d %10.4f %10.4f\n', cat, sum(mask), mae_an, mae_rt);
end
fprintf(fid, '\n');

fprintf(fid, 'Bias de massa total\n');
fprintf(fid, '  Ground-truth: %.2f UEs-equiv.\n', mass_true);
fprintf(fid, '  Analítico:    %.2f UEs-equiv. (bias: %+.0f%%)\n', ...
    mass_analytic, 100*(mass_analytic/mass_true - 1));
fprintf(fid, '  RT:           %.2f UEs-equiv. (bias: %+.0f%%)\n', ...
    mass_rt, 100*(mass_rt/mass_true - 1));
fprintf(fid, '\n');

fprintf(fid, 'Hotspot recovery (top-3, dist. ao pico mais próximo)\n');
fprintf(fid, '  %-25s %10s %10s\n', 'Hotspot', 'Analítico (m)', 'RT (m)');
for h = 1:size(topHotspots,1)
    fprintf(fid, '  hotspot%d (E=%+.0f, N=%+.0f) %10.1f %10.1f\n', ...
        h, topHotspots(h,1), topHotspots(h,2), ...
        dist_hotspot_an(h), dist_hotspot_rt(h));
end
fprintf(fid, '\n');

fprintf(fid, 'Tempo de execução\n');
fprintf(fid, '  rho_true:     %.2f s\n', t_true);
fprintf(fid, '  rho_analytic: %.2f s  (%d iterações MC)\n', t_analytic, N_MC);
fprintf(fid, '  rho_rt:       %.2f s\n', t_rt);

fclose(fid);
fprintf('Métricas salvas: %s\n', outTxt);
fprintf('=== Concluído. ===\n');

%% =========================================================
%  Funções locais auxiliares
%% =========================================================
function draw_basemap_on(ax, bgRGB, mapXlim, mapYlim, hasBasemap)
% Desenha basemap OSM no eixo, ou fundo branco se indisponível.
if hasBasemap
    image(ax, mapXlim, mapYlim, bgRGB);
end
set(ax,'YDir','normal');
axis(ax,'equal');  hold(ax,'on');  grid(ax,'on');
xlim(ax, mapXlim);  ylim(ax, mapYlim);
end

function cmap = red_ramp(N)
% Rampa claro-rosa → vermelho-escuro.
if nargin < 1; N = 256; end
r0 = [0.98 0.85 0.85];   r1 = [0.45 0.00 0.00];
t  = linspace(0,1,N).';
cmap = (1-t).*r0 + t.*r1;
end

function cmap = diverging_bwr(N)
% Azul (negativo) → branco (zero) → vermelho (positivo).
if nargin < 1; N = 256; end
half = ceil(N/2);
blue2white = [(linspace(0,1,half))', (linspace(0,1,half))', ones(half,1)];
white2red  = [ones(N-half,1), (linspace(1,0,N-half))', (linspace(1,0,N-half))'];
cmap = [blue2white; white2red];
end

% ====================================================================
%  simulate_measurements — cópia literal de gnb_density_demo_geo.m
%  NÃO MODIFICAR
% ====================================================================
function ueRaw = simulate_measurements(ueTrue, gnb, params)
K = size(ueTrue,1);
ueRaw = struct('rsrp_dBm',{},'aoa_rad',{});
if K == 0;  return;  end
ueRaw(K).rsrp_dBm = NaN;

xG = gnb.pos(1);  yG = gnb.pos(2);
EIRP = params.Pt_dBm + params.Gt_dB + params.Gr_dB - params.PL0_dB;

for k = 1:K
    dE = ueTrue(k,1) - xG;
    dN = ueTrue(k,2) - yG;
    d  = max(sqrt(dE^2 + dN^2), params.d_min);
    phi_true = atan2(dN, dE);
    ueRaw(k).rsrp_dBm = EIRP - 10*params.n*log10(d) ...
                       + params.sigma_xi_dB * randn();
    ueRaw(k).aoa_rad  = phi_true + params.sigma_phi_rad * randn();
end
end

% ====================================================================
%  local_gnb_build_density_map — cópia literal de gnb_density_demo_geo.m
%  NÃO MODIFICAR
% ====================================================================
function rhoMap = local_gnb_build_density_map(gnb, ueRaw, XG, YG, params)
[Ny, Nx] = size(XG);
xVec = XG(1,:);  yVec = YG(:,1);
rho  = zeros(Ny, Nx);

xG = gnb.pos(1);   yG = gnb.pos(2);
EIRP_dB       = params.Pt_dBm + params.Gt_dB + params.Gr_dB - params.PL0_dB;
ln10_over_10n = log(10) / (10*params.n);
truncSig      = 4;

for k = 1:numel(ueRaw)
    ue = ueRaw(k);
    d_hat = 10.^( (EIRP_dB - ue.rsrp_dBm) / (10*params.n) );
    d_hat = max(params.d_min, min(d_hat, params.d_max));

    if isfield(ue,'aoa_rad') && ~isnan(ue.aoa_rad)
        phi_hat   = ue.aoa_rad;
        sigma_phi = params.sigma_phi_rad;
    else
        phi_hat   = 2*pi*rand();
        sigma_phi = pi;
    end

    x_k = xG + d_hat * cos(phi_hat);
    y_k = yG + d_hat * sin(phi_hat);

    sigma_r = min(ln10_over_10n * d_hat * params.sigma_xi_dB, params.cap_sigma_m);
    sigma_p = min(d_hat * sigma_phi,                          params.cap_sigma_m);

    c = cos(phi_hat);  s = sin(phi_hat);
    R     = [ c -s ;  s  c ];
    Sigma = R * diag([sigma_r^2, sigma_p^2]) * R.';

    [~, Deig] = eig(Sigma);
    smax = sqrt(max(diag(Deig)));   Rtrunc = truncSig * smax;

    ix = find(xVec >= x_k - Rtrunc & xVec <= x_k + Rtrunc);
    iy = find(yVec >= y_k - Rtrunc & yVec <= y_k + Rtrunc);
    if isempty(ix) || isempty(iy);  continue;  end

    Xs = XG(iy,ix);   Ys = YG(iy,ix);
    invS = inv(Sigma);   detS = det(Sigma);
    dxg  = Xs - x_k;     dyg  = Ys - y_k;
    q    = invS(1,1)*dxg.^2 + 2*invS(1,2)*dxg.*dyg + invS(2,2)*dyg.^2;
    g    = exp(-0.5*q) / (2*pi*sqrt(detS));
    rho(iy,ix) = rho(iy,ix) + g;
end

rhoMap = min(rho / params.rho_op, 1);
end
