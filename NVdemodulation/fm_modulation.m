% Execute este script para salvar os resultados ao lado deste arquivo.
%% ============================================================
% NV 10 níveis – evolução temporal com GKSL vetorizado
% Duas fontes de micro-ondas cossenoidais: SIG + REF
%
% Versão com dinâmica coerente nos 10 níveis:
%   - Hamiltoniano estático em g, e, u
%   - termo MW aplicado ao manifold g (ver mw_manifolds abaixo)
%   - canais dissipativos via Lindblad
%   - integração temporal por RK4 + splitting de Strang
%
% REVISÃO (tau_ue = 100 fs):
%   Com dt = 20 ps, Gamma_ue*dt = 200, muito acima do limite de
%   estabilidade do RK4 (|lambda*dt| ~ 2.8). O termo u -> e é separado
%   e integrado EXATAMENTE por splitting de Strang:
%
%       y <- expm(Lue*dt/2) * y
%       y <- RK4( L0 + Lmw(t) , dt ) * y        (parte não rígida)
%       y <- expm(Lue*dt/2) * y
%
%   Os níveis |u> continuam no modelo. A expm é calculada uma única vez.
%   Os geradores MW são pré-calculados (sem kron dentro do laço).
%
% Base (índices):
% 1..3 : |g,+1>,|g,0>,|g,-1>
% 4..6 : |e,+1>,|e,0>,|e,-1>
% 7..9 : |u,+1>,|u,0>,|u,-1>
% 10   : |s>
%% ============================================================

clear; clc; close all;
script_dir = fileparts(mfilename('fullpath'));

%% -------------------------
% 1) Inputs do usuário
%% -------------------------
Bstatic = [0 0 5e-3];     % T

% Tempo total
Ttot = 4e-6;              % s

% Direção da MW
mw_dir = [1 0 0];
mw_dir = mw_dir / norm(mw_dir);

% Manifolds que recebem o acoplamento MW: [g e u]
% (o código original aplicava apenas em g)
mw_manifolds = [true false false];

% Fonte SIG
f_sig   = 3.000e9;        % Hz
S_sig   = 1e8;            % W/m^2
phi_sig = 0;              % rad

% Fonte REF
f_ref   = 3.010e9;        % Hz
S_ref   = 1e8;            % W/m^2
phi_ref = 0;              % rad

% Laser 532 tratado incoerentemente
P532  = 40e-3;            % W
k_532 = 1e10;              % (1/s)/W
R_532 = k_532 * P532;     % 1/s

% Passo temporal
% Obs.: com dt = 20 ps, omega*dt ~ 0.38 nas portadoras de ~3 GHz; o erro
% de fase do RK4 equivale a um deslocamento numérico de ~0.5 MHz na
% ressonância. dt = 10 ps reduz isso ~30x (custo 2x).
dt = 2e-11;               % s
Nt = floor(Ttot/dt) + 1;
t = (0:Nt-1)' * dt;

%% -------------------------
% 2) Taxas
%% -------------------------
% Taxas em s^-1, sem fator 2*pi.
% Radiativo e ISC: Tayefeh Younesi et al. (2025), Tabela I,
% https://doi.org/10.1103/PhysRevApplied.23.054019
Gamma_r = 66e6;
Gamma_isc0 = 7.9e6;
Gamma_isc_pm = 53e6;
Gamma_r0 = Gamma_r;
Gamma_rpm = Gamma_r;

% Saida do singleto por subnivel: medias da Tabela 1 de Tetienne et al.
% (2012), https://doi.org/10.1088/1367-2630/14/10/103033
Gamma_sg0_eq = 0.98e6;
Gamma_sgpm_eq = 0.73e6;
tau_s = 1/(Gamma_sg0_eq + 2*Gamma_sgpm_eq);
p_s_to_g0 = Gamma_sg0_eq*tau_s;

% Tempo vibronico efetivo, motivado pela escala de femtossegundos medida
% por Ulbricht et al. (2018), https://doi.org/10.1103/PhysRevB.97.220302
tau_ue = 100e-15;
Gamma_ue = 1/tau_ue;

nv_rates = struct('Gamma_r', Gamma_r, 'Gamma_isc0', Gamma_isc0, ...
    'Gamma_isc_pm', Gamma_isc_pm, 'Gamma_sg0', Gamma_sg0_eq, ...
    'Gamma_sgpm_each', Gamma_sgpm_eq, 'tau_s', tau_s, 'p_s0', p_s_to_g0, ...
    'tau_ue', tau_ue, 'tau_e0', 1/(Gamma_r + Gamma_isc0), ...
    'tau_epm', 1/(Gamma_r + Gamma_isc_pm));

%% -------------------------
% 3) Conversão MW: S -> B
%% -------------------------
mu0 = 4*pi*1e-7;
c0  = 299792458;

Bsig_amp = sqrt(mu0 * S_sig / c0);   % Tesla
Bref_amp = sqrt(mu0 * S_ref / c0);   % Tesla

%% -------------------------
% 4) Operadores de spin e Hamiltonianos estáticos
%% -------------------------
gamma_e = 2*pi*28e9;      % rad/s/T

Dg = 2*pi*2.87e9;         % rad/s  (ground)
De = 2*pi*1.42e9;         % rad/s  (excited, efetivo)
Du = 2*pi*1.42e9;         % rad/s  (upper, efetivo, sem frequência óptica)
Es = 0;                   % rad/s  (singleto)

Sx = (1/sqrt(2)) * [0 1 0; 1 0 1; 0 1 0];
Sy = (1/sqrt(2)) * [0 -1i 0; 1i 0 -1i; 0 1i 0];
Sz = [1 0 0; 0 0 0; 0 0 -1];

Hspin = @(D, Bvec) D*(Sz*Sz) + gamma_e*(Bvec(1)*Sx + Bvec(2)*Sy + Bvec(3)*Sz);

Hg = Hspin(Dg, Bstatic);
He = Hspin(De, Bstatic);
Hu = Hspin(Du, Bstatic);

d   = 10;
I10 = eye(d);

Hstatic = zeros(d);
Hstatic(1:3,1:3) = Hg;
Hstatic(4:6,4:6) = He;
Hstatic(7:9,7:9) = Hu;
Hstatic(10,10)   = Es;

%% -------------------------
% 5) Índices
%% -------------------------
igp = 1;  ig0 = 2;  igm = 3;
iep = 4;  ie0 = 5;  iem = 6;
iup = 7;  iu0 = 8;  ium = 9;
is  = 10;

lin = @(i) (i-1)*d + i;   % posição de rho(i,i) em y = rho(:)

%% -------------------------
% 6) Operadores de Lindblad
%% -------------------------
L       = {};
Lue_ops = {};             % u -> e separado (parte rígida)

% (i) u -> e spin-conserving
for m = 1:3
    iu = 6 + m;   % 7..9
    ie = 3 + m;   % 4..6
    A = zeros(d);
    A(ie, iu) = sqrt(Gamma_ue);
    Lue_ops{end+1} = A;
end

% (ii) Decaimento radiativo e -> g
A = zeros(d); A(igp, iep) = sqrt(Gamma_rpm); L{end+1} = A;
A = zeros(d); A(ig0, ie0) = sqrt(Gamma_r0 ); L{end+1} = A;
A = zeros(d); A(igm, iem) = sqrt(Gamma_rpm); L{end+1} = A;

% (iii) ISC e -> s
A = zeros(d); A(is, iep) = sqrt(Gamma_isc_pm); L{end+1} = A;
A = zeros(d); A(is, ie0) = sqrt(Gamma_isc0 ); L{end+1} = A;
A = zeros(d); A(is, iem) = sqrt(Gamma_isc_pm); L{end+1} = A;

% (iv) s -> g
A = zeros(d); A(igp, is) = sqrt(Gamma_sgpm_eq); L{end+1} = A;
A = zeros(d); A(ig0, is) = sqrt(Gamma_sg0_eq ); L{end+1} = A;
A = zeros(d); A(igm, is) = sqrt(Gamma_sgpm_eq); L{end+1} = A;

%% -------------------------
% 7) Pumping incoerente g -> u
%% -------------------------
Lp = {};
for m = 1:3
    ig = m;
    iu = 6 + m;
    A = zeros(d);
    A(iu, ig) = sqrt(R_532);
    Lp{end+1} = A;
end

%% -------------------------
% 8) Pré-cálculo dos Liouvillianos
%% -------------------------
dissip = @(A) kron(conj(A), A) - 0.5*kron(I10, A'*A) - 0.5*kron((A'*A).', I10);

% Parte não rígida (vai para o RK4)
allL  = [L(:); Lp(:)];
Ldiss = zeros(d*d, d*d);
for n = 1:numel(allL)
    Ldiss = Ldiss + dissip(allL{n});
end

Lcoh_static = -1i * (kron(I10, Hstatic) - kron(Hstatic.', I10));

L0 = sparse(Ldiss + Lcoh_static);

% Parte rígida: u -> e, propagador exato de meio passo
Lue = zeros(d*d, d*d);
for n = 1:numel(Lue_ops)
    Lue = Lue + dissip(Lue_ops{n});
end
E_ue_half = sparse(expm(Lue*dt/2));

% Geradores MW pré-calculados: Lmw(t) = Bx*GX + By*GY + Bz*GZ
mw_super = @(H) -1i*(kron(I10, H) - kron(H.', I10));
blk_mw   = @(S3) blkdiag(mw_manifolds(1)*S3, mw_manifolds(2)*S3, mw_manifolds(3)*S3, 0);

GX = sparse(mw_super(gamma_e*blk_mw(Sx)));
GY = sparse(mw_super(gamma_e*blk_mw(Sy)));
GZ = sparse(mw_super(gamma_e*blk_mw(Sz)));

% Verificação de estabilidade
lam_rk4 = max(abs(eig(full(L0))));
fprintf('=== ESTABILIDADE ===\n');
fprintf('Gamma_ue*dt (tratado exato)    = %.3g\n', Gamma_ue*dt);
fprintf('|lambda|max*dt (parte em RK4)  = %.3g  (limite ~2.8)\n\n', lam_rk4*dt);
if lam_rk4*dt > 2.5
    error('dt grande demais para a parte nao rigida: reduza dt.');
end

%% -------------------------
% 9) Estado inicial e buffers
%% -------------------------
rho0 = zeros(d);
rho0(ig0, ig0) = 1;   % começa em |g,0>
y = rho0(:);

Pg_p = zeros(Nt,1);  Pg_0 = zeros(Nt,1);  Pg_m = zeros(Nt,1);
Pe_p = zeros(Nt,1);  Pe_0 = zeros(Nt,1);  Pe_m = zeros(Nt,1);
Pu_p = zeros(Nt,1);  Pu_0 = zeros(Nt,1);  Pu_m = zeros(Nt,1);

Ps    = zeros(Nt,1);
Fluor = zeros(Nt,1);
Trrho = zeros(Nt,1);

idx_pop = lin(1:d);   % índices das populações em y

% Campo MW total (vetorizado, fora do laço)
Bmw_scalar = Bsig_amp*cos(2*pi*f_sig*t + phi_sig) + Bref_amp*cos(2*pi*f_ref*t + phi_ref);
Bx_t = Bmw_scalar * mw_dir(1);
By_t = Bmw_scalar * mw_dir(2);
Bz_t = Bmw_scalar * mw_dir(3);

%% -------------------------
% 10) Loop temporal: Strang (u->e exato) + RK4
%% -------------------------
fprintf('Simulando %d passos...\n', Nt);
t_wall = tic;

for k = 1:Nt
    pop = real(y(idx_pop));

    Pg_p(k) = pop(igp);  Pg_0(k) = pop(ig0);  Pg_m(k) = pop(igm);
    Pe_p(k) = pop(iep);  Pe_0(k) = pop(ie0);  Pe_m(k) = pop(iem);
    Pu_p(k) = pop(iup);  Pu_0(k) = pop(iu0);  Pu_m(k) = pop(ium);
    Ps(k)   = pop(is);

    Trrho(k) = sum(pop);

    % Fluorescência total radiativa e -> g
    Fluor(k) = Gamma_rpm*pop(iep) + Gamma_r0*pop(ie0) + Gamma_rpm*pop(iem);

    if k == Nt
        break
    end

    % Liouvilliano do passo (sem kron no laço)
    Lt = L0 + Bx_t(k)*GX + By_t(k)*GY + Bz_t(k)*GZ;

    % Meio passo exato u -> e
    y = E_ue_half * y;

    % RK4 na parte não rígida
    k1 = Lt * y;
    k2 = Lt * (y + 0.5*dt*k1);
    k3 = Lt * (y + 0.5*dt*k2);
    k4 = Lt * (y + dt*k3);
    y  = y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4);

    % Meio passo exato u -> e
    y = E_ue_half * y;

    % Correção leve de hermiticidade/traço e verificação numérica
    if mod(k, 2000) == 0
        rho_tmp = reshape(y, d, d);
        rho_tmp = 0.5*(rho_tmp + rho_tmp');
        tr_tmp  = real(trace(rho_tmp));
        if ~isfinite(tr_tmp) || tr_tmp <= 0
            error('Instabilidade numerica detectada em t = %.4f us', t(k)*1e6);
        end
        y = rho_tmp(:) / tr_tmp;
    end

    if mod(k, 20000) == 0
        fprintf('  %7d/%7d  (%.0f%%)  %.1f s\n', k, Nt, 100*k/Nt, toc(t_wall));
    end
end

fprintf('Simulacao concluida em %.1f s\n\n', toc(t_wall));

%% -------------------------
% 11) Informações gerais
%% -------------------------
f_beat = abs(f_ref - f_sig);

fprintf('f_sig   = %.6f GHz\n', f_sig/1e9);
fprintf('f_ref   = %.6f GHz\n', f_ref/1e9);
fprintf('f_beat  = %.3f MHz\n', f_beat/1e6);
fprintf('Bsig    = %.3e T\n', Bsig_amp);
fprintf('Bref    = %.3e T\n', Bref_amp);
fprintf('Nt      = %d\n', Nt);
fprintf('trace final(rho) = %.12f\n', Trrho(end));
fprintf('Pop. máxima em |u> nos pontos amostrados = %.3e\n', max(Pu_p+Pu_0+Pu_m));

%% -------------------------
% 12) Plots (IEEE Standard Style)
%% -------------------------

% Criar pasta exclusiva para os plots deste código
outDir_freq = fullfile(script_dir, 'resultados_fm');
if ~exist(outDir_freq, 'dir')
    mkdir(outDir_freq);
end

%% -------------------------
% 13) Fluorescence plot
%% -------------------------
fig1 = figure('Color','w','Position',[80 80 1100 650]);
ax1  = axes('Parent',fig1); hold(ax1,'on');

plot(ax1, t*1e6, Fluor, '-', 'LineWidth', 4.6);

box(ax1,'on');
ax1.TickDir     = 'in';
ax1.TickLength  = [0.015 0.015];
ax1.LineWidth   = 1.8;
ax1.FontSize    = 22;
ax1.FontWeight  = 'bold';
ax1.XMinorTick  = 'on';
ax1.YMinorTick  = 'on';
grid(ax1,'on'); ax1.GridAlpha = 0.12;

xlabel(ax1,'Time (\mus)','FontSize',28,'FontWeight','bold');
ylabel(ax1,'Radiative fluorescence (arb.)','FontSize',28,'FontWeight','bold');
title(ax1,'Total fluorescence','FontSize',24,'FontWeight','bold');

exportgraphics(fig1, fullfile(outDir_freq, 'fluorescenceFM.png'), 'Resolution', 600);

%% -------------------------
% 14) Spectral verification of the fluorescence
%% -------------------------
frac_cut = 0.2;
k0 = floor(frac_cut*Nt) + 1;
Fsig_time = Fluor(k0:end);
Fsig_time = Fsig_time - mean(Fsig_time);
Nfft = length(Fsig_time);
w = local_hann(Nfft);
Fwin = Fsig_time .* w;
Y = fft(Fwin);
P2 = abs(Y)/Nfft;
P1 = P2(1:floor(Nfft/2)+1);
P1(2:end-1) = 2*P1(2:end-1);
Fs = 1/dt;
f_axis = Fs*(0:floor(Nfft/2))/Nfft;
search_bw = max(5e6, 0.3*f_beat);
idx_band = find(f_axis >= max(0, f_beat-search_bw) & f_axis <= (f_beat+search_bw));
if isempty(idx_band)
    warning('Beat-frequency search window is empty.');
    f_peak = NaN;
else
    [~, imax_local] = max(P1(idx_band));
    idx_peak = idx_band(imax_local);
    f_peak = f_axis(idx_peak);
end
fprintf('Fluorescence spectral peak = %.3f MHz\n', f_peak/1e6);
fprintf('Expected |f_sig-f_ref|     = %.3f MHz\n', f_beat/1e6);
fprintf('Relative error             = %.3f %%\n', 100*abs(f_peak-f_beat)/f_beat);
fprintf('Resolucao espectral (1/T)  = %.3f MHz\n', 1/(Nfft*dt)/1e6);

fig2 = figure('Color','w','Position',[80 80 1100 650]);
ax2  = axes('Parent',fig2); hold(ax2,'on');

plot(ax2, f_axis/1e6, P1, '-', 'LineWidth', 4.6, 'DisplayName', 'FFT of fluorescence');
xline(ax2, f_beat/1e6, '--r', 'LineWidth', 2.5, 'DisplayName', 'Expected |f_{sig}-f_{ref}|');
if ~isnan(f_peak)
    xline(ax2, f_peak/1e6, '--k', 'LineWidth', 2.5, 'DisplayName', 'Peak found');
end

box(ax2,'on');
ax2.TickDir     = 'in';
ax2.TickLength  = [0.015 0.015];
ax2.LineWidth   = 1.8;
ax2.FontSize    = 22;
ax2.FontWeight  = 'bold';
ax2.XMinorTick  = 'on';
ax2.YMinorTick  = 'on';
grid(ax2,'on'); ax2.GridAlpha = 0.12;

xlabel(ax2,'Frequency (MHz)','FontSize',28,'FontWeight','bold');
ylabel(ax2,'|FFT(Fluor)|','FontSize',28,'FontWeight','bold');
title(ax2,'Fluorescence spectrum','FontSize',24,'FontWeight','bold');
xlim(ax2,[0, max(5*f_beat/1e6, 50)]);

lgd2 = legend(ax2,'Location','northeast');
set(lgd2,'FontSize',20,'FontWeight','bold','Box','on');

exportgraphics(fig2, fullfile(outDir_freq, 'espectroFM.png'), 'Resolution', 600);

% --- Zoomed view around the beat-frequency peak ---
fig3 = figure('Color','w','Position',[80 80 1100 650]);
ax3  = axes('Parent',fig3); hold(ax3,'on');

plot(ax3, f_axis/1e6, P1, '-', 'LineWidth', 4.6, 'DisplayName', 'FFT of fluorescence');
xline(ax3, f_beat/1e6, '--r', 'LineWidth', 2.5, 'DisplayName', 'Expected |f_{sig}-f_{ref}|');
if ~isnan(f_peak)
    xline(ax3, f_peak/1e6, '--k', 'LineWidth', 2.5, 'DisplayName', 'Peak found');
end

box(ax3,'on');
ax3.TickDir     = 'in';
ax3.TickLength  = [0.015 0.015];
ax3.LineWidth   = 1.8;
ax3.FontSize    = 22;
ax3.FontWeight  = 'bold';
ax3.XMinorTick  = 'on';
ax3.YMinorTick  = 'on';
grid(ax3,'on'); ax3.GridAlpha = 0.12;

xlabel(ax3,'Frequency (MHz)','FontSize',28,'FontWeight','bold');
ylabel(ax3,'|FFT(Fluor)|','FontSize',28,'FontWeight','bold');
title(ax3,'Fluorescence spectrum (Zoom)','FontSize',24,'FontWeight','bold');
zoom_bw = max(0.02*f_beat/1e6, 0.02);
xlim(ax3,[f_beat/1e6 - zoom_bw, f_beat/1e6 + zoom_bw]);

lgd3 = legend(ax3,'Location','northeast');
set(lgd3,'FontSize',20,'FontWeight','bold','Box','on');

exportgraphics(fig3, fullfile(outDir_freq, 'espectroFMzoom.png'), 'Resolution', 600);

%% -------------------------
% 15) Time-domain autocorrelation
%% -------------------------
[acf, lags] = local_biased_acf(Fsig_time);
lags_us = lags * dt * 1e6;

fig4 = figure('Color','w','Position',[80 80 1100 650]);
ax4  = axes('Parent',fig4); hold(ax4,'on');

plot(ax4, lags_us, acf, '-', 'LineWidth', 4.6);

box(ax4,'on');
ax4.TickDir     = 'in';
ax4.TickLength  = [0.015 0.015];
ax4.LineWidth   = 1.8;
ax4.FontSize    = 22;
ax4.FontWeight  = 'bold';
ax4.XMinorTick  = 'on';
ax4.YMinorTick  = 'on';
grid(ax4,'on'); ax4.GridAlpha = 0.12;

xlabel(ax4,'Lag (\mus)','FontSize',28,'FontWeight','bold');
ylabel(ax4,'Autocorrelation (biased)','FontSize',28,'FontWeight','bold');
title(ax4,'Autocorrelation of the fluorescence signal','FontSize',24,'FontWeight','bold');
xlim(ax4,[-2, 2]);

exportgraphics(fig4, fullfile(outDir_freq, 'autocorrelacaoFLuorescenciaFM.png'), 'Resolution', 600);

% FFT da autocorrelação
N_acf = length(acf);
Y_acf = fft(acf);
P_acf = abs(Y_acf) / N_acf;
P_acf = P_acf(1:floor(N_acf/2)+1);
P_acf(2:end-1) = 2 * P_acf(2:end-1);
f_acf = Fs * (0:floor(N_acf/2)) / N_acf;

idx_band_acf = find(f_acf >= max(0, f_beat-search_bw) & f_acf <= (f_beat+search_bw));
if ~isempty(idx_band_acf)
    [~, imax_acf] = max(P_acf(idx_band_acf));
    f_peak_acf = f_acf(idx_band_acf(imax_acf));
else
    f_peak_acf = NaN;
end
fprintf('\n--- Autocorrelation FFT ---\n');
fprintf('Spectral peak via autocorrelation = %.3f MHz\n', f_peak_acf/1e6);
fprintf('Relative error                    = %.3f %%\n', 100*abs(f_peak_acf-f_beat)/f_beat);

fig5 = figure('Color','w','Position',[80 80 1100 650]);
ax5  = axes('Parent',fig5); hold(ax5,'on');

plot(ax5, f_acf/1e6, P_acf, '-', 'LineWidth', 4.6, 'DisplayName', 'FFT of the autocorrelation');
xline(ax5, f_beat/1e6, '--r', 'LineWidth', 2.5, 'DisplayName', 'Expected |f_{sig}-f_{ref}|');
if ~isnan(f_peak_acf)
    xline(ax5, f_peak_acf/1e6, '--k', 'LineWidth', 2.5, 'DisplayName', 'Peak found');
end

box(ax5,'on');
ax5.TickDir     = 'in';
ax5.TickLength  = [0.015 0.015];
ax5.LineWidth   = 1.8;
ax5.FontSize    = 22;
ax5.FontWeight  = 'bold';
ax5.XMinorTick  = 'on';
ax5.YMinorTick  = 'on';
grid(ax5,'on'); ax5.GridAlpha = 0.12;

xlabel(ax5,'Frequency (MHz)','FontSize',28,'FontWeight','bold');
ylabel(ax5,'|FFT(ACF)|','FontSize',28,'FontWeight','bold');
title(ax5,'Spectrum via fluorescence autocorrelation','FontSize',24,'FontWeight','bold');
xlim(ax5,[0, max(5*f_beat/1e6, 50)]);

lgd5 = legend(ax5,'Location','northeast');
set(lgd5,'FontSize',20,'FontWeight','bold','Box','on');

exportgraphics(fig5, fullfile(outDir_freq, 'autocorrelacaoEspectroFM.png'), 'Resolution', 600);

%% -------------------------
% Salvar dados para o Python
%% -------------------------
out_FM.t      = t;
out_FM.Fluor  = Fluor;
out_FM.f_sig  = f_sig;
out_FM.f_ref  = f_ref;
out_FM.f_beat = f_beat;
out_FM.dt     = dt;
out_FM.f_peak = f_peak;
out_FM.f_peak_acf = f_peak_acf;
out_FM.nv_rates = nv_rates;
out_FM.R_532 = R_532;
save(fullfile(outDir_freq, 'out_FM.mat'), 'out_FM');

function w = local_hann(n)
    if n == 1
        w = 1;
    else
        w = 0.5 - 0.5*cos(2*pi*(0:n-1)'/(n-1));
    end
end

function [acf, lags] = local_biased_acf(signal)
    n = numel(signal);
    nfft = 2^nextpow2(2*n - 1);
    power = abs(fft(signal, nfft)).^2;
    positive = real(ifft(power))/n;
    acf = [flipud(positive(2:n)); positive(1:n)];
    lags = (-n+1:n-1)';
end