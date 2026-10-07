function [imgRGB, xLim, yLim] = osm_basemap(centerLat, centerLon, halfX, halfY, zoom, varargin)
%OSM_BASEMAP  Download OpenStreetMap tiles and stitch a local-NED basemap.
%
%   [IMG, XLIM, YLIM] = OSM_BASEMAP(LAT, LON, HALFX, HALFY, ZOOM) returns
%   a true-color image IMG whose extent in the local NED-horizontal frame
%   (origin at the supplied lat/lon, North->+X, East->+Y, units: metres)
%   is XLIM(1)..XLIM(2) by YLIM(1)..YLIM(2). The image is built by
%   downloading the OSM tiles that cover the requested window, projecting
%   them with the standard Web-Mercator math, then resampling onto a
%   regular metre grid. Cached on disk so repeated calls hit local files.
%
%   Inputs:
%       LAT, LON     centre of the local frame, decimal degrees (WGS-84)
%       HALFX, HALFY half-extent of the window in metres (North, East)
%       ZOOM         OSM zoom level (15-17 for corridor-scale views)
%
%   Name-value:
%       'TileServer' default 'https://tile.openstreetmap.org/%d/%d/%d.png'
%       'UserAgent'  default 'matlab-uam-sim/1.0 (research use)'
%
%   Render with:
%       image(yLim, xLim, permute(imgRGB,[2 1 3]));
%       set(gca,'YDir','normal');
%
%   See also: gnb_density_demo_geo.

p = inputParser;
addParameter(p,'TileServer','https://tile.openstreetmap.org/%d/%d/%d.png');
addParameter(p,'UserAgent','matlab-uam-sim/1.0 (research use)');
parse(p,varargin{:});
TILE_URL = p.Results.TileServer;
UA       = p.Results.UserAgent;

cacheDir = fullfile(tempdir,'matlab_osm_cache');
if ~exist(cacheDir,'dir'); mkdir(cacheDir); end

%% 1) Local-NED window -> bounding lat/lon ------------------------------
R   = 6378137;                              % WGS-84 equatorial radius
mPerDegLat = pi*R/180;
mPerDegLon = mPerDegLat * cosd(centerLat);

latN = centerLat + halfX / mPerDegLat;      % North edge   (largest lat)
latS = centerLat - halfX / mPerDegLat;      % South edge   (smallest lat)
lonW = centerLon - halfY / mPerDegLon;      % West edge    (smallest lon)
lonE = centerLon + halfY / mPerDegLon;      % East edge    (largest lon)

%% 2) Tile coverage at the requested zoom -------------------------------
%   OSM/Web-Mercator convention: y increases southward, so the NORTH edge
%   of the window maps to the SMALLEST y tile index and the SOUTH edge to
%   the LARGEST. Compute fractional indices first, then round outward.
[xt_W_frac, yt_N_frac] = latlon2tile_frac(latN, lonW, zoom);   % NW corner
[xt_E_frac, yt_S_frac] = latlon2tile_frac(latS, lonE, zoom);   % SE corner

xt_min = floor(min(xt_W_frac, xt_E_frac));
xt_max = floor(max(xt_W_frac, xt_E_frac));
yt_min = floor(min(yt_N_frac, yt_S_frac));
yt_max = floor(max(yt_N_frac, yt_S_frac));

% Guard: if any axis collapsed to a single tile, force a 1-tile-wide
% range. Should be impossible at zoom 16 with halfX = 1600 m, but the
% earlier crash showed we cannot trust it without an assertion.
nMax = 2^zoom - 1;
xt_min = max(0,    min(xt_min, nMax));
xt_max = max(0,    min(xt_max, nMax));
yt_min = max(0,    min(yt_min, nMax));
yt_max = max(0,    min(yt_max, nMax));
if xt_max < xt_min;  xt_max = xt_min;  end
if yt_max < yt_min;  yt_max = yt_min;  end

xtRange = xt_min:xt_max;
ytRange = yt_min:yt_max;

fprintf('[osm_basemap] zoom=%d, x-tiles=%d..%d (%d), y-tiles=%d..%d (%d)\n', ...
    zoom, xt_min, xt_max, numel(xtRange), yt_min, yt_max, numel(ytRange));

%% 3) Download and stitch ----------------------------------------------
tileSize = 256;
W = numel(xtRange) * tileSize;
H = numel(ytRange) * tileSize;
if W < 2 || H < 2
    error('osm_basemap:emptyMosaic', ...
        'Mosaic collapsed to %dx%d pixels. Check zoom/halfX/halfY.', W, H);
end
mosaic = uint8(255*ones(H, W, 3));

opts = weboptions('UserAgent', UA, 'Timeout', 30);
for ix = 1:numel(xtRange)
    for iy = 1:numel(ytRange)
        xt = xtRange(ix);  yt = ytRange(iy);
        fname = sprintf('osm_z%d_x%d_y%d.png', zoom, xt, yt);
        fpath = fullfile(cacheDir, fname);
        if ~exist(fpath,'file')
            url = sprintf(TILE_URL, zoom, xt, yt);
            try
                websave(fpath, url, opts);
            catch ME
                warning('osm_basemap:tileDownload', ...
                    'Tile (%d,%d,%d) failed: %s. Leaving white.', ...
                    zoom, xt, yt, ME.message);
                continue;
            end
        end
        try
            tile = imread(fpath);
            if size(tile,3) == 1; tile = repmat(tile,[1 1 3]); end
            r = (iy-1)*tileSize + (1:tileSize);
            c = (ix-1)*tileSize + (1:tileSize);
            mosaic(r, c, :) = tile(:,:,1:3);
        catch ME
            warning('osm_basemap:tileRead', ...
                'Could not read cached tile %s: %s', fname, ME.message);
        end
    end
end

%% 4) Map the mosaic pixel grid to local metres ------------------------
% Sample a regular metre grid inside the requested window (1 m per pixel)
nx = max(2, round(2*halfX));               % North dimension
ny = max(2, round(2*halfY));               % East  dimension
xMetreVec = linspace(-halfX, +halfX, nx);  % North
yMetreVec = linspace(-halfY, +halfY, ny);  % East
[YM, XM]  = meshgrid(yMetreVec, xMetreVec);

latSample = centerLat + XM / mPerDegLat;
lonSample = centerLon + YM / mPerDegLon;

% Convert each (lat,lon) sample to fractional pixel coords in the mosaic
[xt_s, yt_s] = latlon2tile_frac(latSample, lonSample, zoom);
colSample = (xt_s - xt_min) * tileSize + 0.5;   % +0.5 = pixel center
rowSample = (yt_s - yt_min) * tileSize + 0.5;

[Hm, Wm, ~] = size(mosaic);
colSample = max(1, min(Wm, colSample));
rowSample = max(1, min(Hm, rowSample));

% interp2 wants V(row,col) sampled at (Xq,Yq) where Xq=col, Yq=row
imgRGB = zeros(nx, ny, 3, 'uint8');
for ch = 1:3
    Vch = double(mosaic(:,:,ch));
    imgRGB(:,:,ch) = uint8( ...
        interp2(Vch, colSample, rowSample, 'linear', 255) );
end

xLim = [-halfX, +halfX];
yLim = [-halfY, +halfY];
end


% ----------------------------------------------------------------------
%  Web-Mercator helpers (matching OSM tile convention)
% ----------------------------------------------------------------------
function [xtf, ytf] = latlon2tile_frac(lat, lon, z)
n   = 2^z;
xtf = (lon + 180) / 360 * n;
latRad = lat * pi/180;
ytf = (1 - log(tan(latRad) + 1./cos(latRad))/pi) / 2 * n;
end

function [lat, lon] = tile2latlon(xt, yt, z)  %#ok<DEFNU>
n = 2^z;
lon    = xt / n * 360 - 180;
latRad = atan(sinh(pi*(1 - 2*yt/n)));
lat    = latRad * 180/pi;
end