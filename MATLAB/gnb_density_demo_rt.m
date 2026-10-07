function gnb_density_demo_rt()
%GNB_DENSITY_DEMO_RT  Demo interativo com medidas reais do Sionna RT.
%
%   Versão adaptada de gnb_density_demo_geo.m que carrega medidas
%   pré-computadas pelo Sionna RT (ray-tracing físico da cena Interlagos)
%   em vez de gerar medidas sintéticas a cada redraw.
%
%   Diferenças em relação ao original:
%     • UEs carregados de output/measurements_rt.mat (posições fixas).
%     • Botões "Add random" e drag de UE removidos.
%     • redraw chama load_measurements em vez de simulate_measurements.
%     • O painel esquerdo mostra os UEs reais coloridos por path_type.
%
%   Mantidos intactos: layout, basemap OSM, estimador
%   local_gnb_build_density_map, colormap, alpha, controles de sigma.
%
%   Requer: osm_basemap.m e load_measurements.m no path.

here = fileparts(mfilename('fullpath'));
addpath(here);
addpath(fullfile(here, '..', 'MATLAB'));   % garante load_measurements no path

if ~exist('osm_basemap','file')
    error('gnb_density_demo_rt:missingDep', ...
        'osm_basemap.m deve estar no path MATLAB.');
end

%% =========================================================
%  Constantes geográficas e de grade (idênticas ao demo original)
%% =========================================================
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

%% gNB no frame MATLAB: East=0, North=0 (gNB Sionna foi subtraído no .mat)
gnb = struct('pos',[0, 0, -62], 'cellId',1);   % Down=-62 → Up=62 m AGL

%% Parâmetros do estimador (default do demo — não tunar)
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

%% =========================================================
%  Carrega medidas RT (posições verdadeiras e medidas RSRP/AoA)
%% =========================================================
fprintf('Carregando medidas Sionna RT...\n');
% Chama load_measurements sem argumentos úteis — retorna todos os UEs
ueAll = load_measurements([], gnb, params);
N = numel(ueAll);
fprintf('  %d UEs carregados.\n', N);

% Extrai posições verdadeiras para o painel esquerdo
ueTrue = zeros(N, 2);
for k = 1:N
    ueTrue(k, 1) = ueAll(k).pos_east;
    ueTrue(k, 2) = ueAll(k).pos_north;
end

%% Cores por path_type para o painel de ground-truth
PATH_TYPES  = {'LoS','reflected','diffracted','transmitted','none'};
PATH_COLORS = [0.18 0.80 0.44;   % LoS      — verde
               0.20 0.60 0.87;   % reflected — azul
               0.90 0.49 0.13;   % diffracted — laranja
               0.61 0.35 0.71;   % transmitted — roxo
               0.50 0.50 0.50];  % none       — cinza

ptColors = zeros(N, 3);
for k = 1:N
    idx = find(strcmp(PATH_TYPES, ueAll(k).path_type), 1);
    if ~isempty(idx)
        ptColors(k,:) = PATH_COLORS(idx,:);
    else
        ptColors(k,:) = [0.5 0.5 0.5];
    end
end

%% =========================================================
%  Basemap OSM
%% =========================================================
fprintf('Baixando tiles OSM para (%.5f, %.5f)... ', centerLat, centerLon);
[bgRGB, ~, ~] = osm_basemap(centerLat, centerLon, halfNorth, halfEast, zoomLvl);
fprintf('OK.\n');

%% =========================================================
%  Layout da figura
%% =========================================================
fig = figure('Color','w','Position',[60 60 1500 760], ...
    'Name','gNB Density Demo — Sionna RT (Interlagos)', ...
    'NumberTitle','off');

axL = axes('Parent',fig,'Position',[0.04 0.22 0.44 0.72]);
axR = axes('Parent',fig,'Position',[0.52 0.22 0.44 0.72]);

drawBasemap(axL, bgRGB, mapXlim, mapYlim);
drawBasemap(axR, bgRGB, mapXlim, mapYlim);

%% Painel esquerdo: ground truth com UEs coloridos por path_type
title(axL, sprintf('Ground truth Sionna RT  (%d UEs)', N));
xlabel(axL,'East (m)');  ylabel(axL,'North (m)');

hold(axL,'on');
plot(axL, gnb.pos(1), gnb.pos(2), 'r^','MarkerSize',16, ...
     'MarkerFaceColor','r','LineWidth',1.5);
text(axL, gnb.pos(1)+30, gnb.pos(2)+30, 'gNB', ...
     'FontWeight','bold','Color','r');

% Pontos UE coloridos por path_type
hUE = scatter(axL, ueTrue(:,1), ueTrue(:,2), 18, ptColors, 'filled', ...
              'MarkerEdgeColor','none', 'MarkerFaceAlpha',0.7);

% Legenda path_type
for ti = 1:numel(PATH_TYPES)
    mask = strcmp({ueAll.path_type}, PATH_TYPES{ti});
    if ~any(mask); continue; end
    plot(axL, NaN, NaN, 'o','Color',PATH_COLORS(ti,:), ...
         'MarkerFaceColor',PATH_COLORS(ti,:),'MarkerSize',6, ...
         'DisplayName', sprintf('%s (%d)', PATH_TYPES{ti}, sum(mask)));
end
legend(axL,'show','Location','northeast','FontSize',7);

%% Painel direito: densidade estimada
title(axR,'$\bar{\rho}(x,t)$ — estimado com RSRP + AoA (Sionna RT)', ...
      'Interpreter','latex');
xlabel(axR,'East (m)');  ylabel(axR,'North (m)');

hold(axR,'on');
hImg = imagesc(axR, xVec, yVec, zeros(size(XG)));
set(hImg,'AlphaData', zeros(size(XG)));
caxis(axR,[0 1]);
colormap(axR, red_ramp(256));
cb = colorbar(axR);
cb.Label.String = '$\bar{\rho} \in [0,1]$';
cb.Label.Interpreter = 'latex';

plot(axR, gnb.pos(1), gnb.pos(2), '^','MarkerSize',16, ...
     'MarkerFaceColor','w','MarkerEdgeColor','k','LineWidth',1.2);

linkaxes([axL axR],'xy');

%% =========================================================
%  Barra de controles
%% =========================================================
% Botão: recarrega (útil após mudar sigma via slider)
uicontrol(fig,'Style','pushbutton','String','Reestimar', ...
    'Units','pixels','Position',[40 30 100 30], ...
    'Callback',@(s,e) redraw(fig));

uicontrol(fig,'Style','text','String','sigma_xi (dB):', ...
    'Position',[160 26 90 22],'BackgroundColor','w', ...
    'HorizontalAlignment','right','FontWeight','bold');
uicontrol(fig,'Style','slider','Min',0.5,'Max',12,'Value',6, ...
    'Position',[255 30 150 22], ...
    'Callback',@(s,e) onSlider(fig,'sigma_xi_dB',s.Value));

uicontrol(fig,'Style','text','String','sigma_phi (deg):', ...
    'Position',[420 26 100 22],'BackgroundColor','w', ...
    'HorizontalAlignment','right','FontWeight','bold');
uicontrol(fig,'Style','slider','Min',0.5,'Max',30,'Value',3, ...
    'Position',[525 30 150 22], ...
    'Callback',@(s,e) onSlider(fig,'sigma_phi_deg',s.Value));

uicontrol(fig,'Style','text','String','heatmap alpha:', ...
    'Position',[690 26 100 22],'BackgroundColor','w', ...
    'HorizontalAlignment','right','FontWeight','bold');
uicontrol(fig,'Style','slider','Min',0.0,'Max',1.0,'Value',0.65, ...
    'Position',[795 30 110 22], ...
    'Callback',@(s,e) onSlider(fig,'alpha',s.Value));

hStatus = uicontrol(fig,'Style','text', ...
    'String', sprintf('%d UEs RT   |   calculando...', N), ...
    'Position',[920 28 550 24],'BackgroundColor','w', ...
    'HorizontalAlignment','left','FontWeight','bold');

%% =========================================================
%  Estado da figura (guidata)
%% =========================================================
data.gnb      = gnb;
data.params   = params;
data.alpha    = 0.65;
data.XG       = XG;    data.YG    = YG;
data.xVec     = xVec;  data.yVec  = yVec;
data.mapXlim  = mapXlim;  data.mapYlim = mapYlim;
data.cellSize = cellSize;
data.ueTrue   = ueTrue;    % posições verdadeiras (N×2)
data.ueAll    = ueAll;     % struct array completo (inclui medidas RT)
data.axL      = axL;   data.axR   = axR;
data.hUE      = hUE;   data.hImg  = hImg;
data.hStatus  = hStatus;
guidata(fig, data);

redraw(fig);
end


% ====================================================================
%  Redraw — usa load_measurements (não simulate_measurements)
% ====================================================================
function redraw(fig)
data = guidata(fig);

% load_measurements retorna todos os UEs RT (ignora ueTrue)
ueRaw  = load_measurements(data.ueTrue, data.gnb, data.params);
rhoMap = local_gnb_build_density_map(data.gnb, ueRaw, ...
            data.XG, data.YG, data.params);

set(data.hImg, 'CData', rhoMap, 'AlphaData', data.alpha * rhoMap);

N     = numel(ueRaw);
integ = sum(rhoMap(:)) * data.params.rho_op * data.cellSize^2;
set(data.hStatus,'String', sprintf( ...
    '%d UEs RT   |   massa integrada: %.2f   (target: %d)', ...
    N, integ, N));
drawnow limitrate;
end


% ====================================================================
%  Callbacks de slider
% ====================================================================
function onSlider(fig, fieldName, val)
data = guidata(fig);
switch fieldName
    case 'sigma_xi_dB';   data.params.sigma_xi_dB   = val;
    case 'sigma_phi_deg'; data.params.sigma_phi_rad  = deg2rad(val);
    case 'alpha';         data.alpha                 = val;
end
guidata(fig,data);  redraw(fig);
end


% ====================================================================
%  Helpers de basemap e colormap (idênticos ao demo original)
% ====================================================================
function drawBasemap(ax, bgRGB, mapXlim, mapYlim)
image(ax, mapXlim, mapYlim, bgRGB);
set(ax,'YDir','normal');
axis(ax,'equal');  hold(ax,'on');  grid(ax,'on');
xlim(ax, mapXlim);  ylim(ax, mapYlim);
end

function cmap = red_ramp(N)
if nargin < 1; N = 256; end
r0 = [0.98 0.85 0.85];
r1 = [0.45 0.00 0.00];
t  = linspace(0,1,N).';
cmap = (1-t).*r0 + t.*r1;
end


% ====================================================================
%  Estimador — cópia do original (NÃO modificar)
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
