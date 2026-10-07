function ueRaw = load_measurements(ueTrue, gnb, params) %#ok<INUSD>
%LOAD_MEASUREMENTS  Drop-in de simulate_measurements usando dados Sionna RT.
%
%   Assinatura idêntica a simulate_measurements(ueTrue, gnb, params),
%   mas IGNORA ueTrue, gnb e params — a função retorna medidas reais
%   pré-computadas pelo Sionna RT 2.x (ray-tracing físico).
%
%   O arquivo measurements_rt.mat é carregado uma única vez (persistent
%   cache) e o struct array ueRaw é reconstruído a cada chamada a partir
%   dos arrays flat armazenados no .mat.
%
%   Campos do struct ueRaw(k):
%     .rsrp_dBm  — potência recebida (dBm)
%     .aoa_rad   — azimute de chegada CCW do +East (rad), frame MATLAB
%     .pos_east  — East verdadeiro (m, relativo à gNB)
%     .pos_north — North verdadeiro (m, relativo à gNB)
%     .path_type — categoria do caminho dominante (string)

persistent cache;

if isempty(cache)
    % Localiza measurements_rt.mat relativo a este arquivo .m
    here    = fileparts(mfilename('fullpath'));
    matFile = fullfile(here, '..', 'output', 'measurements_rt.mat');
    if ~exist(matFile, 'file')
        error('load_measurements:fileNotFound', ...
            'Arquivo não encontrado: %s\nRode: python scripts/export_to_matlab.py', ...
            matFile);
    end
    cache = load(matFile);
    fprintf('[load_measurements] %d UEs Sionna RT carregados de:\n  %s\n', ...
        int32(cache.n_ues), matFile);
end

N = int32(cache.n_ues);

% Reconstrói struct array a partir dos arrays flat do .mat
ueRaw = repmat(struct('rsrp_dBm', NaN, 'aoa_rad', NaN, ...
                      'pos_east', NaN, 'pos_north', NaN, ...
                      'path_type', ''), 1, N);

for k = 1:N
    ueRaw(k).rsrp_dBm  = cache.rsrp_dBm(k);
    ueRaw(k).aoa_rad   = cache.aoa_rad(k);
    ueRaw(k).pos_east  = cache.pos_east(k);
    ueRaw(k).pos_north = cache.pos_north(k);
    ueRaw(k).path_type = cache.path_type{k};
end

end
