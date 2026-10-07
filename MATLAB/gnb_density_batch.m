% gnb_density_batch.m — Gera mapa de densidade RT sem GUI (modo batch).
%
% Uso: matlab -batch "run('MATLAB/gnb_density_batch.m')"
%      (executar a partir do diretório raiz do projeto)
%
% Saída: output/gnb_density_rt_preview.png

fprintf('=== gnb_density_batch.m ===\n');

% Garante path correto
here = fileparts(mfilename('fullpath'));
if isempty(here); here = pwd; end
addpath(here);

%% Parâmetros — idênticos ao demo
centerLat = -(23 + 42/60 + 06.9/3600);
centerLon = -(46 + 42/60 + 03.4/3600);

halfEast  = 1600;
halfNorth = 600;
mapXlim   = [-halfEast,  +halfEast];
mapYlim   = [-halfNorth, +halfNorth];
zoomLvl   = 16;
cellSize  = 50;

xVec = (mapXlim(1)+cellSize/2):cellSize:(mapXlim(2)-cellSize/2);
yVec = (mapYlim(1)+cellSize/2):cellSize:(mapYlim(2)-cellSize/2);
[XG, YG] = meshgrid(xVec, yVec);

gnb = struct('pos',[0, 0, -62], 'cellId',1);

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

%% Carrega medidas RT
fprintf('Carregando medidas Sionna RT...\n');
ueRaw  = load_measurements([], gnb, params);
N = numel(ueRaw);
fprintf('  %d UEs carregados.\n', N);

%% Posições verdadeiras
ueTrue = zeros(N, 2);
for k = 1:N
    ueTrue(k, 1) = ueRaw(k).pos_east;
    ueTrue(k, 2) = ueRaw(k).pos_north;
end

%% Calcula mapa de densidade
fprintf('Calculando rho_map...\n');
t0 = tic;
rhoMap = local_gnb_build_density_map(gnb, ueRaw, XG, YG, params);
fprintf('  rho_map em %.1f s | max=%.4f | integral=%.2f\n', ...
    toc(t0), max(rhoMap(:)), ...
    sum(rhoMap(:)) * params.rho_op * cellSize^2);

%% Basemap OSM
fprintf('Baixando tiles OSM...\n');
try
    [bgRGB, ~, ~] = osm_basemap(centerLat, centerLon, halfNorth, halfEast, zoomLvl);
    hasBasemap = true;
catch ME
    fprintf('  Aviso: OSM indisponível (%s) — fundo branco.\n', ME.message);
    hasBasemap = false;
end

%% Figura em modo offscreen
fig = figure('Visible','off','Color','w','Position',[0 0 1400 720]);

axL = subplot(1,2,1, 'Parent', fig);
axR = subplot(1,2,2, 'Parent', fig);

%% Colormap por path_type
PATH_TYPES  = {'LoS','reflected','diffracted','transmitted','none'};
PATH_COLORS = [0.18 0.80 0.44;
               0.20 0.60 0.87;
               0.90 0.49 0.13;
               0.61 0.35 0.71;
               0.50 0.50 0.50];

ptColors = zeros(N, 3);
for k = 1:N
    idx = find(strcmp(PATH_TYPES, ueRaw(k).path_type), 1);
    if ~isempty(idx); ptColors(k,:) = PATH_COLORS(idx,:);
    else;             ptColors(k,:) = [0.5 0.5 0.5]; end
end

%% Painel esquerdo: ground truth
if hasBasemap
    image(axL, mapXlim, mapYlim, bgRGB);
    set(axL,'YDir','normal');
end
hold(axL,'on');  grid(axL,'on');
xlim(axL, mapXlim);  ylim(axL, mapYlim);
axis(axL,'equal');
scatter(axL, ueTrue(:,1), ueTrue(:,2), 10, ptColors, 'filled', ...
        'MarkerFaceAlpha',0.7,'MarkerEdgeColor','none');
plot(axL, 0, 0, 'r^','MarkerSize',14,'MarkerFaceColor','r','LineWidth',1.5);
text(axL, 30, 30, 'gNB','FontWeight','bold','Color','r','FontSize',9);

% Legenda
for ti = 1:numel(PATH_TYPES)
    mask = strcmp({ueRaw.path_type}, PATH_TYPES{ti});
    if ~any(mask); continue; end
    plot(axL, NaN, NaN, 'o','Color',PATH_COLORS(ti,:), ...
         'MarkerFaceColor',PATH_COLORS(ti,:),'MarkerSize',7, ...
         'DisplayName', sprintf('%s (%d)', PATH_TYPES{ti}, sum(mask)));
end
legend(axL,'show','Location','northeast','FontSize',7);
title(axL, sprintf('Ground truth Sionna RT  (%d UEs)', N),'FontSize',10);
xlabel(axL,'East (m)');  ylabel(axL,'North (m)');

%% Painel direito: mapa de densidade
if hasBasemap
    image(axR, mapXlim, mapYlim, bgRGB);
    set(axR,'YDir','normal');
end
hold(axR,'on');  grid(axR,'on');
xlim(axR, mapXlim);  ylim(axR, mapYlim);
axis(axR,'equal');

hImg = imagesc(axR, xVec, yVec, rhoMap);
set(hImg, 'AlphaData', 0.65 * rhoMap);
caxis(axR, [0 1]);
colormap(axR, red_ramp(256));
cb = colorbar(axR);
cb.Label.String = 'rho normalizado';

plot(axR, 0, 0, '^','MarkerSize',14,'MarkerFaceColor','w', ...
     'MarkerEdgeColor','k','LineWidth',1.2);

integ = sum(rhoMap(:)) * params.rho_op * cellSize^2;
title(axR, sprintf('\\rho(x) estimado — RSRP + AoA (Sionna RT)  |  massa=%.1f / %d', ...
    integ, N), 'FontSize',10);
xlabel(axR,'East (m)');  ylabel(axR,'North (m)');

%% Salva PNG
outDir = fullfile(here, '..', 'output');
outFile = fullfile(outDir, 'gnb_density_rt_preview.png');
print(fig, outFile, '-dpng', '-r150');
fprintf('Imagem salva: %s\n', outFile);
close(fig);

%% ====================================================================
%  Estimador local (cópia — não modificar)
%% ====================================================================
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

%% Colormap auxiliar
function cmap = red_ramp(N)
if nargin < 1; N = 256; end
r0 = [0.98 0.85 0.85];  r1 = [0.45 0.00 0.00];
t  = linspace(0,1,N).';
cmap = (1-t).*r0 + t.*r1;
end
