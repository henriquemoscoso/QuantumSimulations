clear; clc; close all
script_dir = fileparts(mfilename('fullpath'));

%% =========================================================================
%  simular_qam.m — 64-QAM com laser contínuo e taxas ópticas revisadas
%
%  Demodulador 64-QAM baseado em fluorescência de centros NV.
%  Versão SEM LOCK-IN COERENTE — sem necessidade de sincronia de fase.
%
%  ALTERAÇÕES (v2 e v3):
%
%  1. INTEGRAÇÃO EXATA (expm aumentada / Van Loan)
%     L_cw é constante dentro de cada amostra, então:
%
%        expm([L 0; wF' 0]*Tsamp) = [ e^{L*Tsamp}            0 ]
%                                   [ int_0^Tsamp wF'e^{Ls}ds 1 ]
%
%     A expm dá a propagação de rho e a fluorescência integrada, ambas
%     exatas para o gerador fixo usado no passo. Incondicionalmente estável, o que
%     permite tau_ue = 100 fs. Não há mais dt nem sub-passos.
%
%  2. TEMPORIZAÇÃO CONSISTENTE
%     N1 = fs_cw/fIF e Nsamp_sym = Tsym/Tsamp são DERIVADOS (antes N1 era
%     fixo em 10 e a janela de demodulação não coincidia com o símbolo).
%     O guard interval é múltiplo inteiro de um ciclo IF, para que a
%     janela integrada sempre feche ciclos inteiros e o termo conjugado
%     s*·e^{i2·phi_IF} seja rejeitado.
%
%  3. ESTIMADOR DE FASE/FREQUÊNCIA DATA-AIDED POR FFT
%     Substitui unwrap(angle(z/s)), que é frágil nos pontos internos da
%     64-QAM. Cada símbolo de treino é ponderado por |s|^2.
%
%  4. EQUALIZADOR NORMALIZADO
%     z é normalizado para rms = 1 antes da equalização e o teste de
%     singularidade passa a ser relativo (|a|^2 - |b|^2 < 1e-3·|a|^2).
%
%  5. SENSIBILIDADES calculadas a partir de s_hat (pós-equalização), na
%     escala da constelação; o ganho físico é recuperado com z_scale.
%
%  6. OPÇÃO DE ELIMINAÇÃO ADIABÁTICA dos níveis |u> (d = 7, ~8x mais
%     rápido). Padrão: modelo completo com |u> (d = 10).
%
%  7. (v3) CACHE DE PROPAGADORES: N1 expm por símbolo em vez de uma por
%     amostra (~50x menos expm), mantendo o modelo completo com os níveis
%     |u> (tau_ue = 100 fs) e a integração exata por expm aumentada.
%
%  Demodulação robusta:
%  1. Laser verde CW sempre ligado.
%  2. Fluorescência integrada por janela de amostragem.
%  3. Ruído físico na fluorescência.
%  4. Demodulação I/Q sem referência de fase (fase local = 0).
%  5. Estimação de offset de fase/frequência via treinamento (FFT).
%  6. Equalização widely-linear: z = a*s + b*conj(s) + c.
%  7. Refinamento decision-directed.
% =========================================================================

%% =========================================================================
% 0) Opções de modelo
% =========================================================================

modelo_adiabatico = false;   % true: elimina |u> (bombeio g->e direto, d = 7)
                             % false: modelo completo com |u> (d = 10)

%% =========================================================================
% 1) Parâmetros de temporização (todos derivados de fs_cw, fIF e N2)
% =========================================================================

fs_cw  = 200e6;            % taxa de amostragem CW [Hz]  (>= 20 x fIF)
Tsamp  = 1/fs_cw;          % período de amostragem [s]

fIF    = 1e6;             % frequência intermediária [Hz]
N2     = 50;               % ciclos IF por símbolo  -> Tsym = 50 us

N1     = round(fs_cw/fIF); % amostras por ciclo IF (derivado)
Rsym   = fIF / N2;         % taxa de símbolos [baud]
Tsym   = 1/Rsym;           % duração de um símbolo [s]

Nsamp_sym = round(Tsym/Tsamp);   % amostras por símbolo (derivado)

assert(abs(fs_cw/fIF - round(fs_cw/fIF)) < 1e-9, 'fs_cw/fIF nao e inteiro');
assert(abs(Tsym/Tsamp - round(Tsym/Tsamp)) < 1e-9, 'Tsym/Tsamp nao e inteiro');
assert(N1 >= 10, 'Menos de 10 amostras por ciclo IF: aumente fs_cw.');
assert(Nsamp_sym == N1*N2, 'Inconsistencia: Nsamp_sym ~= N1*N2.');

fprintf('[TIMING — SEM LOCK-IN]\n');
fprintf('  fs_cw         = %.1f MHz\n', fs_cw/1e6);
fprintf('  fIF           = %.3f MHz\n', fIF/1e6);
fprintf('  N1            = %d amostras/ciclo IF\n', N1);
fprintf('  N2            = %d ciclos IF/simbolo\n', N2);
fprintf('  Rsym          = %.2f kHz\n', Rsym/1e3);
fprintf('  Tsym          = %.2f us\n', Tsym*1e6);
fprintf('  Nsamp/sym     = %d\n', Nsamp_sym);

if Tsym < 2e-6
    warning(['Tsym = %.2f us e curto frente a memoria do NV (~0.25 us com tau_s = 410 ns). ' ...
             'Risco de interferencia entre simbolos: aumente N2.'], Tsym*1e6);
end

%% =========================================================================
% 1b) Parâmetros de erro de fase/frequência no TX
%     (o receptor NADA sabe sobre esses valores)
% =========================================================================

phi0_tx          = 1.23;       % fase inicial desconhecida [rad]
freq_offset_ppm  = 5.0;        % offset de frequência TX vs RX [ppm de fIF]
phase_drift_Hz   = 0.0;        % deriva de fase linear [Hz]

delta_fIF_tx = fIF * (1 + freq_offset_ppm*1e-6);

fprintf('\n[ERRO DE FASE/FREQUENCIA NO TX (receptor nao sabe)]\n');
fprintf('  phi0_tx          = %.4f rad\n', phi0_tx);
fprintf('  freq_offset_ppm  = %.2f ppm  (= %.2f Hz)\n', freq_offset_ppm, fIF*freq_offset_ppm*1e-6);
fprintf('  phase_drift_Hz   = %.2f Hz\n', phase_drift_Hz);

%% =========================================================================
% 2) Constantes físicas do centro NV
% =========================================================================

gamma_e = 2*pi * 28e9;       % razão giromagnética [rad/s/T]
D_gs    = 2*pi * 2.87e9;     % zero-field splitting [rad/s]
Bz      = 5e-3;              % campo magnético estático [T]

omega_01 = D_gs + gamma_e * Bz;
f_sig    = omega_01/(2*pi);
f_ref    = f_sig + fIF;

fprintf('\n[FISICA NV]\n');
fprintf('  f_01          = %.6f GHz\n', f_sig/1e9);
fprintf('  f_ref         = %.6f GHz\n', f_ref/1e9);
fprintf('  Bz            = %.2f mT\n', Bz*1e3);

%% =========================================================================
% 3) Taxas dissipativas
% =========================================================================

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

%% =========================================================================
% 4) Laser CW 532 nm
% =========================================================================

% Com as taxas acima, a PL satura pelo gargalo do singleto (tau_s = 410 ns):
% meia saturação em R_sat ~ 1e7 /s. Com R = 3e8 (~30 R_sat) o contraste para
% Omega/2pi = 1.6 MHz cai para ~0.3 %; com R ~ 1.5 R_sat ele fica em ~23 %
% e o tempo de resposta (~0.25 us) continua << Tsym.
R532_cw = 3e8;             % taxa de bombeio óptico [1/s] (era 3e8)

fprintf('\n[LASER 532 nm]\n');
fprintf('  R532_cw = %.2e /s\n', R532_cw);

%% =========================================================================
% 5) Parâmetros de micro-ondas
% =========================================================================

% Preserva a amplitude do experimento numérico original ao alterar as taxas.
Omega_sig0 = 1e7;            % rad/s
Omega_ref0 = Omega_sig0;

fprintf('\n[MW]\n');
fprintf('  Omega_sig0/(2pi) = %.3f MHz\n', Omega_sig0/(2*pi*1e6));
fprintf('  Omega_ref0/(2pi) = %.3f MHz\n', Omega_ref0/(2*pi*1e6));

%% =========================================================================
% 6) Constelação 64-QAM
% =========================================================================

M_qam = 64;
sqrtM = round(sqrt(M_qam));

lev = -(sqrtM-1):2:(sqrtM-1);

[II, QQ] = meshgrid(lev, lev);
const = II(:) + 1i*QQ(:);
const = const / sqrt(mean(abs(const).^2));

Tcal       = 4e-6;
Nrep_train = 10;                       % era 5 (192 símbolos já sobram para 3 coeficientes)
Ntrain_sym = Nrep_train * M_qam;
Ndata_sym  = 3000;

rng(42);

train_idx = repmat((1:M_qam)', Nrep_train, 1);
data_idx  = randi([1 M_qam], Ndata_sym, 1);

sym_idx = [train_idx; data_idx];
s_sym   = const(sym_idx);

Nsym = length(s_sym);

Tmsg = Nsym * Tsym;
Ttot = Tcal + Tmsg;

fprintf('\n[QAM]\n');
fprintf('  M_qam       = %d\n', M_qam);
fprintf('  Ntrain      = %d simbolos\n', Ntrain_sym);
fprintf('  Ndata       = %d simbolos\n', Ndata_sym);
fprintf('  Nsym total  = %d simbolos\n', Nsym);

%% =========================================================================
% 7) Modelo de ruído experimental
% =========================================================================

noise_enable = false;

N_NV_eff      = 1e7;
eta_det       = 0.4;
R_dark        = 1e3;
sigma_read_ct = 0.1;

rin_rms  = 3e-4;
rin_tau  = 20e-6;

rng_noise = 2026;

%% =========================================================================
% 7.1) Pasta de saída
% =========================================================================

if noise_enable
    outDir = fullfile(script_dir, 'resultados_qam');
    figTag = 'semLockin_com_ruido';
else
    outDir = fullfile(script_dir, 'resultados_qam_sem_ruido');
    figTag = 'semLockin_sem_ruido';
end

if ~exist(outDir, 'dir')
    mkdir(outDir);
end

fprintf('\n[SAIDA]\n');
fprintf('  Graficos serao salvos em: %s\n', outDir);

%% =========================================================================
% 8) Espaço de Hilbert
% =========================================================================

if modelo_adiabatico
    d   = 7;
    igp = 1;  ig0 = 2;  igm = 3;
    iep = 4;  ie0 = 5;  iem = 6;
    is_ = 7;
else
    d   = 10;
    igp = 1;  ig0 = 2;  igm = 3;
    iep = 4;  ie0 = 5;  iem = 6;
    iup = 7;  iu0 = 8;  ium = 9;
    is_ = 10;
end

I_d = eye(d);

fprintf('\n[MODELO]\n');
if modelo_adiabatico
    fprintf('  Eliminacao adiabatica de |u> (d = %d)\n', d);
else
    fprintf('  Modelo completo com |u>, tau_ue = %.0f fs (d = %d)\n', tau_ue*1e15, d);
end

%% =========================================================================
% 9) Hamiltoniano estático
% =========================================================================

Delta_p = 0;
Delta_m = -2*gamma_e*Bz;

Hstatic = zeros(d);
Hstatic(igp,igp) = Delta_p;
Hstatic(igm,igm) = Delta_m;

Lcoh_static = -1i*(kron(I_d,Hstatic) - kron(Hstatic.',I_d));

%% =========================================================================
% 10) Operadores de dissipação
% =========================================================================

Lops = {};

if ~modelo_adiabatico
    % u -> e (relaxação vibrônica rápida)
    A = zeros(d); A(iep,iup) = sqrt(Gamma_ue); Lops{end+1} = A;
    A = zeros(d); A(ie0,iu0) = sqrt(Gamma_ue); Lops{end+1} = A;
    A = zeros(d); A(iem,ium) = sqrt(Gamma_ue); Lops{end+1} = A;
end

% e -> g radiativo
A = zeros(d); A(igp,iep) = sqrt(Gamma_rpm); Lops{end+1} = A;
A = zeros(d); A(ig0,ie0) = sqrt(Gamma_r0);  Lops{end+1} = A;
A = zeros(d); A(igm,iem) = sqrt(Gamma_rpm); Lops{end+1} = A;

% e -> s (ISC)
A = zeros(d); A(is_,iep) = sqrt(Gamma_isc_pm); Lops{end+1} = A;
A = zeros(d); A(is_,ie0) = sqrt(Gamma_isc0);  Lops{end+1} = A;
A = zeros(d); A(is_,iem) = sqrt(Gamma_isc_pm); Lops{end+1} = A;

% s -> g
A = zeros(d); A(igp,is_) = sqrt(Gamma_sgpm_eq); Lops{end+1} = A;
A = zeros(d); A(ig0,is_) = sqrt(Gamma_sg0_eq);  Lops{end+1} = A;
A = zeros(d); A(igm,is_) = sqrt(Gamma_sgpm_eq); Lops{end+1} = A;

Ldiss = zeros(d^2);
for k = 1:length(Lops)
    Ldiss = Ldiss + lindblad_superop(Lops{k}, d);
end

%% =========================================================================
% 11) Bombeio óptico CW
% =========================================================================

Lpump_ops = {};

if modelo_adiabatico
    % bombeio direto g -> e
    A = zeros(d); A(iep,igp) = sqrt(R532_cw); Lpump_ops{end+1} = A;
    A = zeros(d); A(ie0,ig0) = sqrt(R532_cw); Lpump_ops{end+1} = A;
    A = zeros(d); A(iem,igm) = sqrt(R532_cw); Lpump_ops{end+1} = A;
else
    % bombeio g -> u
    A = zeros(d); A(iup,igp) = sqrt(R532_cw); Lpump_ops{end+1} = A;
    A = zeros(d); A(iu0,ig0) = sqrt(R532_cw); Lpump_ops{end+1} = A;
    A = zeros(d); A(ium,igm) = sqrt(R532_cw); Lpump_ops{end+1} = A;
end

Ldiss_pump = zeros(d^2);
for k = 1:length(Lpump_ops)
    Ldiss_pump = Ldiss_pump + lindblad_superop(Lpump_ops{k}, d);
end

L_base_cw = Ldiss + Ldiss_pump + Lcoh_static;

%% =========================================================================
% 12) Operadores MW
% =========================================================================

A_raise = zeros(d); A_raise(igp,ig0) = 0.5;
A_lower = zeros(d); A_lower(ig0,igp) = 0.5;

G_raise = -1i*(kron(I_d,A_raise) - kron(A_raise.',I_d));
G_lower = -1i*(kron(I_d,A_lower) - kron(A_lower.',I_d));

Gx_mw = G_raise + G_lower;
Gy_mw = -1i*(G_raise - G_lower);

%% =========================================================================
% 13) Simulação temporal — PROPAGADOR EXATO COM expm AUMENTADA
% =========================================================================

Nsamp_total = round(Ttot/Tsamp);
Nsamp_cal   = round(Tcal/Tsamp);

samp_F = zeros(Nsamp_total,1);
samp_t = ((0:Nsamp_total-1)' + 0.5)*Tsamp;

% Base real exata da parte acessível da matriz densidade. O drive
% mistura apenas g,0 e g,+1 e todos os saltos são de posto 1.
% Assim, somente as populações e Re/Im de rho(g,0; g,+1) evoluem.
idx_pop = sub2ind([d,d], 1:d, 1:d);
idx_01 = sub2ind([d,d], ig0, igp);
idx_10 = sub2ind([d,d], igp, ig0);
lift = zeros(d^2,d+2);
lift(idx_pop,1:d) = eye(d);
lift([idx_01,idx_10],d+1) = [1;1];
lift([idx_01,idx_10],d+2) = [1i;-1i];
restrict = diag([ones(1,d), 0.5, 0.5])*lift';
L_base_cw = real(restrict*L_base_cw*lift);
Gx_mw = real(restrict*Gx_mw*lift);
Gy_mw = real(restrict*Gy_mw*lift);

n2 = d+2;
y = zeros(n2,1);
y(ig0) = 1;

% Fluorescência instantânea; a última linha acumula a integral em cada bin.
wF = zeros(n2,1);
wF([iep,ie0,iem]) = Gamma_r;
M_aug = zeros(n2+1);
M_aug(end,1:n2) = wF.';

fprintf('\n[SIMULACAO — expm aumentada]\n');
fprintf('  Tsamp          = %.2f ns\n', Tsamp*1e9);
fprintf('  Nsamp_total    = %d\n', Nsamp_total);
fprintf('  Tempo fisico   = %.2f us\n', Ttot*1e6);
fprintf('  Dimensao expm  = %d x %d\n', n2+1, n2+1);
fprintf('  Iniciando integracao...\n');

t_wall0 = tic;

assert(Nsamp_total == Nsamp_cal + Nsym*Nsamp_sym, 'Nsamp_total inconsistente.');

% -------------------------------------------------------------------------
% CACHE DE PROPAGADORES POR BLOCO
% Como fs_cw/fIF = N1 é inteiro e s_k é constante dentro do símbolo, os
% Nsamp_sym propagadores de um símbolo são apenas N1 matrizes distintas,
% repetidas N2 vezes. Calculam-se N1 expm por bloco (calibração ou símbolo)
% em vez de uma por amostra. A fase de Omega_ref é exata no início de cada
% bloco; dentro dele, o offset de frequência é congelado (erro impresso
% abaixo; zero quando freq_offset_ppm = 0 e phase_drift_Hz = 0).
% -------------------------------------------------------------------------
eps_f   = delta_fIF_tx/fIF - 1;
err_phi = 2*pi*abs(eps_f)*max(Nsamp_sym, Nsamp_cal)/N1;
fprintf('  Cache de propagadores: %d expm por bloco, %d blocos (%d expm no total)\n', ...
    N1, Nsym+1, N1*(Nsym+1));
fprintf('  Erro de fase maximo por congelar o offset no bloco = %.2e rad\n', err_phi);

maps = zeros(n2+1,n2,N1);  % propagador e linha de integração por fase no IF

for b = 0:Nsym            % b = 0: calibração; b >= 1: símbolo b

    if b == 0
        n_first = 1;                                 Nblk = Nsamp_cal;  sk = 0;
    else
        n_first = Nsamp_cal + (b-1)*Nsamp_sym + 1;   Nblk = Nsamp_sym;  sk = s_sym(b);
    end

    % --- N1 propagadores distintos do bloco ---
    t_blk = (n_first-1)*Tsamp;
    for j = 1:N1
        t0 = t_blk + (j-1)*Tsamp;

        % Fase da referência no TX (desconhecida pelo receptor)
        phi_IF_tx = 2*pi*delta_fIF_tx*t0 + phi0_tx ...
                  + 2*pi*phase_drift_Hz*t0^2 / (2*Ttot);

        Omega_tot = Omega_ref0*exp(1i*phi_IF_tx) + Omega_sig0*sk;

        L_cw = L_base_cw + real(Omega_tot)*Gx_mw + imag(Omega_tot)*Gy_mw;

        % Propagação exata + integral exata da fluorescência (Van Loan)
        M_aug(1:n2,1:n2) = L_cw;
        E = expm(M_aug*Tsamp);

        maps(:,:,j) = E(:,1:n2);
    end

    % --- Propagação do bloco reutilizando os N1 propagadores ---
    for m = 0:Nblk-1
        j  = mod(m, N1) + 1;
        ns = n_first + m;

        integrated = maps(:,:,j)*y;
        samp_F(ns) = integrated(end);
        y = integrated(1:n2);

        if j == N1                       % normaliza uma vez por ciclo IF
            y = y/sum(y(1:d));
        end
    end
    y = y/sum(y(1:d));

    if any(~isfinite(y))
        error('Instabilidade numerica no bloco %d.', b);
    end

    if b > 0 && mod(b, max(1,round(Nsym/20))) == 0
        elapsed = toc(t_wall0);
        pct     = 100*b/Nsym;
        fprintf('  %5d/%5d simbolos  %.0f%%  ETA %.1f s\n', b, Nsym, pct, elapsed/pct*(100-pct));
    end
end

fprintf('  Simulacao concluida em %.1f s\n', toc(t_wall0));

assert(all(isfinite(samp_F)), 'samp_F contem NaN/Inf.');

%% =========================================================================
% 14) Aplicação do ruído na fluorescência
% =========================================================================

samp_F_ideal = samp_F;

if noise_enable

    rng(rng_noise);

    a_rin = exp(-Tsamp/rin_tau);

    rin_white  = randn(Nsamp_total,1);
    gain_noise = zeros(Nsamp_total,1);

    for k = 2:Nsamp_total
        gain_noise(k) = a_rin*gain_noise(k-1) + sqrt(1-a_rin^2)*rin_white(k);
    end

    gain_noise = rin_rms * gain_noise;
    gain_noise = max(gain_noise, -0.95);

    mu_signal_ct = N_NV_eff * eta_det .* samp_F_ideal .* (1 + gain_noise);
    mu_signal_ct = max(mu_signal_ct, 0);

    mu_dark_ct = max(R_dark*Tsamp, 0);

    if exist('poissrnd','file') == 2 && license('test','Statistics_Toolbox')
        counts_signal = poissrnd(mu_signal_ct);
        counts_dark   = poissrnd(mu_dark_ct, Nsamp_total, 1);
    else
        counts_signal = max(round(mu_signal_ct + sqrt(max(mu_signal_ct,0)).*randn(Nsamp_total,1)), 0);
        if mu_dark_ct < 0.05
            counts_dark = double(rand(Nsamp_total,1) < mu_dark_ct);
        else
            counts_dark = max(round(mu_dark_ct + sqrt(mu_dark_ct).*randn(Nsamp_total,1)), 0);
        end
    end

    counts_read  = sigma_read_ct * randn(Nsamp_total,1);
    counts_total = counts_signal + counts_dark + counts_read;
    counts_total = counts_total - mu_dark_ct;

    samp_F = counts_total ./ (N_NV_eff * eta_det);
    clear rin_white gain_noise mu_signal_ct counts_signal counts_dark counts_read counts_total

    fprintf('\n[RUIDO APLICADO]\n');
    fprintf('  SNR bruto = %.2f dB\n', ...
        20*log10(std(samp_F_ideal)/max(std(samp_F - samp_F_ideal),eps)));

else
    fprintf('\n[RUIDO] Desativado.\n');
end

%% =========================================================================
% 15) Demodulação SEM LOCK-IN COERENTE
% =========================================================================

fprintf('\n[DEMODULACAO SEM LOCK-IN]\n');

idx_dc = (samp_t >= Tcal/2) & (samp_t < Tcal);
assert(sum(idx_dc) >= 5, 'Poucas amostras de calibracao.');

win_dc = max(5*Nsamp_sym, 101);
if mod(win_dc,2) == 0; win_dc = win_dc + 1; end

F_slow = movmedian(samp_F, win_dc);
F_ac   = samp_F - F_slow;

fprintf('  janela movmedian    = %d amostras\n', win_dc);

% Misturador local: fase zero, frequência nominal
fIF_local = fIF;
% O oscilador é avaliado nas amostras usadas em cada símbolo.

% Guard interval em número INTEIRO de ciclos IF
guard_frac = 0.15;
Nguard     = N1 * round(guard_frac*Nsamp_sym/N1);
Nintegr    = Nsamp_sym - 2*Nguard;
assert(Nintegr > 10, 'Nintegr muito pequeno.');

w_int = ones(Nintegr,1) / Nintegr;

% ---- Diagnóstico de temporização ----
fprintf('\n[DIAGNOSTICO DE TEMPORIZACAO]\n');
fprintf('  fs_cw/fIF             = %g  (ideal >= 20)\n', fs_cw/fIF);
fprintf('  N1                    = %d\n', N1);
fprintf('  Nsamp_sym             = %d\n', Nsamp_sym);
fprintf('  amostras reais/simb.  = %g\n', Tsym/Tsamp);
fprintf('  ciclos IF na janela   = %g  (precisa ser inteiro)\n', Nintegr*Tsamp*fIF);
fprintf('  ciclos IF no guard    = %g  (precisa ser inteiro)\n', Nguard*Tsamp*fIF);

assert(abs(Nintegr*Tsamp*fIF - round(Nintegr*Tsamp*fIF)) < 1e-6, ...
    'Janela de integracao nao cobre numero inteiro de ciclos IF.');

z_sym = zeros(Nsym,1);
valid = false(Nsym,1);

for ks = 1:Nsym

    n_start = Nsamp_cal + (ks-1)*Nsamp_sym + 1;
    n_end   = Nsamp_cal + ks*Nsamp_sym;

    if n_end > Nsamp_total; break; end

    idx_win = (n_start + Nguard):(n_end - Nguard);
    if length(idx_win) ~= Nintegr; continue; end

    oscillator = exp(1i*2*pi*fIF_local*samp_t(idx_win));
    z_sym(ks) = dot(w_int, F_ac(idx_win).*oscillator);
    valid(ks) = true;
end

z_sym     = z_sym(valid);
s_tx      = s_sym(valid);
sym_idx_v = sym_idx(valid);

Nsym_v   = length(s_tx);
Ntrain_v = min(sum(valid(1:Ntrain_sym)), Ntrain_sym);

fprintf('\n  Simbolos validos = %d\n', Nsym_v);
fprintf('  Treinamento      = %d\n', Ntrain_v);

%% =========================================================================
% 15b) Estimação de offset de fase/frequência — data-aided por FFT
%
%  r_k = z_k * conj(s_k) remove a modulação (ponderando por |s_k|^2).
%  O pico da FFT de r_k dá o offset de frequência residual; a fase do
%  produto escalar com a exponencial estimada dá o offset de fase.
%  phi_offset inclui o atraso de fase do NV em fIF, logo NÃO precisa
%  coincidir com phi0_tx.
% =========================================================================

phase_corr_enable = true;
t_all = (0:Nsym_v-1)' * Tsym;

if phase_corr_enable && Ntrain_v >= 20

    fprintf('\n[CORRECAO DE FASE — estimador data-aided por FFT]\n');

    z_tr_raw = z_sym(1:Ntrain_v);
    s_tr_raw = s_tx(1:Ntrain_v);
    t_tr     = t_all(1:Ntrain_v);

    r = z_tr_raw .* conj(s_tr_raw);

    Nfft_ph = 2^nextpow2(64*Ntrain_v);
    R_ph    = fft(r, Nfft_ph);
    [~, ik] = max(abs(R_ph));
    f_grid  = (0:Nfft_ph-1)' / (Nfft_ph*Tsym);
    f_grid(f_grid >= Rsym/2) = f_grid(f_grid >= Rsym/2) - Rsym;
    df_est  = f_grid(ik);

    phi_offset = angle(sum(r .* exp(-1i*2*pi*df_est*t_tr)));

    fprintf('  offset de freq. estimado = %.3f Hz  (TX: %.3f Hz)\n', ...
        df_est, fIF*freq_offset_ppm*1e-6);
    fprintf('  phi_offset estimado      = %.4f rad\n', phi_offset);

    z_sym_corr = z_sym .* exp(-1i*(2*pi*df_est*t_all + phi_offset));

else
    z_sym_corr = z_sym;
    fprintf('\n[CORRECAO DE FASE] Nao aplicada.\n');
end

%% =========================================================================
% 16) Equalização widely-linear (com z normalizado)
%     z = a*s + b*conj(s) + c
% =========================================================================

% Normalização: rms(z) = 1 (condicionamento numérico)
z_scale    = sqrt(mean(abs(z_sym_corr).^2));
z_sym_corr = z_sym_corr / z_scale;

fprintf('\n[NORMALIZACAO]\n');
fprintf('  z_scale (rms de z antes da normalizacao) = %.4e\n', z_scale);

z_tr = z_sym_corr(1:Ntrain_v);
s_tr = s_tx(1:Ntrain_v);

X_tr = [s_tr, conj(s_tr), ones(Ntrain_v,1)];

lambda_eq = 1e-6 * trace(X_tr'*X_tr) / size(X_tr,2);

keep  = true(Ntrain_v,1);
theta = zeros(3,1);

for it = 1:5

    Xk = X_tr(keep,:);
    zk = z_tr(keep);

    theta = (Xk'*Xk + lambda_eq*eye(3)) \ (Xk'*zk);

    z_fit = X_tr*theta;
    resid = z_tr - z_fit;

    med_r   = median(resid);
    mad_r   = median(abs(resid - med_r));
    sigma_r = 1.4826*mad_r + eps;

    keep_new = abs(resid - med_r) < 4.0*sigma_r;
    if sum(keep_new) > 0.70*Ntrain_v
        keep = keep_new;
    end
end

a_eq = theta(1);
b_eq = theta(2);
c_eq = theta(3);

fprintf('\n[EQUALIZACAO WIDELY-LINEAR]\n');
fprintf('  a_eq       = %+.4e %+.4ei\n', real(a_eq), imag(a_eq));
fprintf('  b_eq       = %+.4e %+.4ei\n', real(b_eq), imag(b_eq));
fprintf('  c_eq       = %+.4e %+.4ei\n', real(c_eq), imag(c_eq));
fprintf('  |a|        = %.4e\n', abs(a_eq));
fprintf('  |b|/|a|    = %.4f  (proximo de 1 => imagem conjugada nao rejeitada)\n', ...
    abs(b_eq)/max(abs(a_eq),eps));
fprintf('  treino usado = %d/%d\n', sum(keep), Ntrain_v);

den_eq = abs(a_eq)^2 - abs(b_eq)^2;

if abs(den_eq) < 1e-3*abs(a_eq)^2

    warning('Equalizador widely-linear quase singular (|b| ~ |a|). Usando equalizador simples.');
    A_simple    = [s_tr, ones(Ntrain_v,1)];
    coef_simple = (A_simple'*A_simple + 1e-6*eye(2)) \ (A_simple'*z_tr);
    a_eq  = coef_simple(1);
    b_eq  = 0;
    c_eq  = coef_simple(2);
    s_hat = (z_sym_corr - c_eq)/a_eq;

else

    zc    = z_sym_corr - c_eq;
    s_hat = (conj(a_eq)*zc - b_eq*conj(zc)) / den_eq;

end

s_hat_tr  = s_hat(1:Ntrain_v);
evm_train = sqrt(mean(abs(s_hat_tr - s_tr).^2) / mean(abs(s_tr).^2))*100;

fprintf('  EVM treino = %.2f %%\n', evm_train);

%% =========================================================================
% 17) Decisão ML inicial
% =========================================================================

D2_all = abs(bsxfun(@minus, s_hat, const.')).^2;

[D2_sort, idx_sort] = sort(D2_all, 2);

idx_dec     = idx_sort(:,1);
s_dec       = const(idx_dec);
conf_metric = D2_sort(:,1) ./ max(D2_sort(:,2), eps);

%% =========================================================================
% 18) Refinamento decision-directed
% =========================================================================

idx_data_all  = (Ntrain_v+1):Nsym_v;
idx_conf_data = idx_data_all(conf_metric(idx_data_all) < 0.20);

fprintf('\n[DECISION-DIRECTED]\n');
fprintf('  Simbolos confiaveis = %d\n', length(idx_conf_data));

if length(idx_conf_data) > 20

    s_ref_dd = [s_tx(1:Ntrain_v); s_dec(idx_conf_data)];
    z_ref_dd = [z_sym_corr(1:Ntrain_v); z_sym_corr(idx_conf_data)];

    X_dd      = [s_ref_dd, conj(s_ref_dd), ones(length(s_ref_dd),1)];
    lambda_dd = 1e-6 * trace(X_dd'*X_dd) / size(X_dd,2);

    theta_dd = (X_dd'*X_dd + lambda_dd*eye(3)) \ (X_dd'*z_ref_dd);

    a_dd   = theta_dd(1);
    b_dd   = theta_dd(2);
    c_dd   = theta_dd(3);
    den_dd = abs(a_dd)^2 - abs(b_dd)^2;

    if abs(den_dd) >= 1e-3*abs(a_dd)^2
        a_eq = a_dd; b_eq = b_dd; c_eq = c_dd;

        zc    = z_sym_corr - c_eq;
        s_hat = (conj(a_eq)*zc - b_eq*conj(zc)) / den_dd;

        D2_all = abs(bsxfun(@minus, s_hat, const.')).^2;
        [~, idx_dec] = min(D2_all, [], 2);
        s_dec = const(idx_dec);

        fprintf('  Refinamento aplicado. novo |b|/|a| = %.4f\n', ...
            abs(b_eq)/max(abs(a_eq),eps));
    else
        fprintf('  Refinamento ignorado: quase singular.\n');
    end

else
    fprintf('  Poucos simbolos confiaveis. Refinamento ignorado.\n');
end

%% =========================================================================
% 19) Métricas finais
% =========================================================================

if Nsym_v > Ntrain_v
    idx_data = (Ntrain_v+1):Nsym_v;
else
    idx_data = 1:Nsym_v;
    warning('Sem dados fora do treinamento.');
end

s_hat_data = s_hat(idx_data);
s_tx_data  = s_tx(idx_data);

idx_true_d = sym_idx_v(idx_data);
idx_dec_d  = idx_dec(idx_data);

EVM = sqrt(mean(abs(s_hat_data - s_tx_data).^2) / mean(abs(s_tx_data).^2))*100;
SER = mean(idx_dec_d ~= idx_true_d)*100;

noise_resid = s_hat_data - s_tx_data;

fprintf('\n=== RESULTADOS FINAIS ===\n');
fprintf('  EVM  = %.2f %%\n', EVM);
fprintf('  SER  = %.2f %%\n', SER);
fprintf('  SNR_eq aproximado = %.2f dB\n', -20*log10(EVM/100));
fprintf('  std erro I = %.4f\n', std(real(noise_resid)));
fprintf('  std erro Q = %.4f\n', std(imag(noise_resid)));

%% =========================================================================
% 19b) Sensibilidade de fase do sensor NV
%
%  η = σ / (∂C_fl/∂φ) / sqrt(F_s)     [°/√Hz]
%
%  σ é estimado a partir de s_hat (pós-equalização, mesma escala de s_tx):
%       ε_k = s_hat(k)/s_tx(k) − 1     (k ∈ treinamento)
% =========================================================================

fprintf('\n=== SENSIBILIDADE DO SENSOR NV ===\n');

idx_train_sens = 1:Ntrain_v;
s_tr_sens      = s_tx(idx_train_sens);
s_hat_tr_sens  = s_hat(idx_train_sens);

valid_sens = abs(s_tr_sens) > 1e-9;
eps_k      = s_hat_tr_sens(valid_sens) ./ s_tr_sens(valid_sens) - 1;

sigma_fl = ( std(real(eps_k)) + std(imag(eps_k)) ) / 2;

fprintf('  σ (desvio padrão relativo, pós-EQ)    = %.4e\n', sigma_fl);

idx_msg  = samp_t >= Tcal;
F_ac_msg = F_ac(idx_msg);
F_dc_msg = mean(samp_F(idx_msg));

C_fl0_est = max(abs(F_ac_msg)) / max(abs(F_dc_msg), eps);

phi_op_deg = 40;
dCfl_dphi  = C_fl0_est * abs(sin(phi_op_deg * pi/180)) * (pi/180);

fprintf('  C_fl0 estimado (contraste máx AC/DC)  = %.4e\n', C_fl0_est);
fprintf('  φ_op (ponto de operação)              = %d°\n',  phi_op_deg);
fprintf('  ∂C_fl/∂φ                              = %.4e /°\n', dCfl_dphi);

Fs_eff = Rsym;
fprintf('  F_s efetivo (= Rsym)                  = %.2f kHz\n', Fs_eff/1e3);

eta_deg  = sigma_fl / dCfl_dphi / sqrt(Fs_eff);
eta_mrad = eta_deg * pi/180 * 1e3;

fprintf('  ──────────────────────────────────────────\n');
fprintf('  η = %.4f °/√Hz  =  %.4f mrad/√Hz\n', eta_deg, eta_mrad);

fprintf('  EVM teórico (= σ × 100)               = %.3f %%\n', sigma_fl*100);
fprintf('  EVM simulado                          = %.3f %%\n', EVM);

N2_vec  = [10 20 40 80 160 320];
eta_vec = zeros(size(N2_vec));
for ki = 1:length(N2_vec)
    Rs_k        = fIF / N2_vec(ki);
    eta_vec(ki) = sigma_fl / dCfl_dphi / sqrt(Rs_k);
end

fig_sens = figure('Name','Sensibilidade vs N2','Position',[200 200 560 380]);
semilogy(N2_vec, eta_vec, 'o-b', 'LineWidth', 1.6, 'MarkerSize', 7);
hold on;
semilogy(N2, eta_deg, 'rs', 'MarkerSize', 12, 'LineWidth', 2.2);
grid on;
xlabel('N_2 (ciclos IF por símbolo)');
ylabel('\eta [°/\surdHz]');
title('Sensibilidade de fase do sensor NV vs integração por símbolo');
legend('η(N_2)', sprintf('Atual N_2=%d  →  η=%.3f°/√Hz', N2, eta_deg), ...
    'Location','northwest');
save_current_figure(fig_sens, outDir, ['08_sensibilidade_' figTag]);

%% =========================================================================
% 19c) Sensibilidade de amplitude do sensor NV
%
%  σ_A: resíduo radial pós-equalização (escala da constelação).
%  Conversão para unidades físicas: z_físico = z_scale · z_normalizado,
%  logo o ganho constelação -> fluorescência é |a_eq|·z_scale.
% =========================================================================

fprintf('\n=== SENSIBILIDADE DE AMPLITUDE DO SENSOR NV ===\n');

valid_A  = abs(s_tr_sens) > 1e-9;
delta_k  = s_hat_tr_sens(valid_A) - s_tr_sens(valid_A);
s_unit_k = s_tr_sens(valid_A) ./ abs(s_tr_sens(valid_A));

delta_A_k  = real( delta_k .* conj(s_unit_k) );
delta_ph_k = imag( delta_k .* conj(s_unit_k) );

sigma_A  = std(delta_A_k);
sigma_ph = std(delta_ph_k);

fprintf('  σ_A  (resíduo radial,  pós-EQ) = %.4e  [unid. constelação]\n', sigma_A);
fprintf('  σ_φ  (resíduo angular, pós-EQ) = %.4e  [unid. constelação]\n', sigma_ph);
fprintf('  σ_A / σ_φ = %.4f  (≈1 se ruído isotrópico)\n', sigma_A/max(sigma_ph,eps));

dCfl_dA = C_fl0_est * abs(cos(phi_op_deg * pi/180));
fprintf('  ∂C_fl/∂A                       = %.4e\n', dCfl_dA);

gain_eq = abs(a_eq) * z_scale;    % constelação -> fluorescência integrada

eta_A_norm = sigma_A * gain_eq / dCfl_dA / sqrt(Fs_eff);

fprintf('  |a_eq|·z_scale (ganho físico)  = %.4e\n', gain_eq);
fprintf('  η_A (ampl. QAM normalizada)    = %.6e /√Hz\n', eta_A_norm);

eta_Rabi_Hz = eta_A_norm * Omega_sig0 / (2*pi);
fprintf('  η_Rabi                         = %.4f Hz/√Hz\n', eta_Rabi_Hz);

eta_B_T  = eta_A_norm * Omega_sig0 * sqrt(2) / gamma_e;
eta_B_nT = eta_B_T * 1e9;

fprintf('  η_B (campo MW)                 = %.4e T/√Hz  =  %.4f nT/√Hz\n', eta_B_T, eta_B_nT);

fprintf('\n  ── Comparação das sensibilidades ──────────────────\n');
fprintf('  η_fase      = %.4f °/√Hz   = %.4f mrad/√Hz\n', eta_deg,  eta_mrad);
fprintf('  η_amplitude = %.6e /√Hz\n', eta_A_norm);
fprintf('  η_Rabi      = %.4f Hz/√Hz\n', eta_Rabi_Hz);
fprintf('  η_B         = %.4f nT/√Hz\n', eta_B_nT);

% Elipse de incerteza IQ
fig_elipse = figure('Name','Elipse de incerteza IQ','Position',[760 200 560 500]);

[~, idx_max_amp] = max(abs(const));
s_ref_pt    = const(idx_max_amp);
tol_clust   = 0.15;
idx_cluster = idx_data( abs(s_hat(idx_data) - s_ref_pt) < tol_clust );

if length(idx_cluster) >= 10
    pts       = s_hat(idx_cluster);
    delta_cl  = pts - s_ref_pt;
    s_unit_cl = s_ref_pt / abs(s_ref_pt);
    sc = 3 * std(real(delta_cl .* conj(s_unit_cl)));
    sp = 3 * std(imag(delta_cl .* conj(s_unit_cl)));
    plot(real(pts), imag(pts), 'b.', 'MarkerSize', 9); hold on;
    plot(real(s_ref_pt), imag(s_ref_pt), 'r+', 'MarkerSize', 14, 'LineWidth', 2.5);
    s_tang = 1i * s_unit_cl;
    quiver(real(s_ref_pt), imag(s_ref_pt), real(s_unit_cl)*sc, imag(s_unit_cl)*sc, 0, ...
        'r', 'LineWidth', 1.8, 'MaxHeadSize', 0.6);
    quiver(real(s_ref_pt), imag(s_ref_pt), real(s_tang)*sp, imag(s_tang)*sp, 0, ...
        'g', 'LineWidth', 1.8, 'MaxHeadSize', 0.6);
    legend('Símbolos recebidos','Ponto ideal', ...
        sprintf('3\\sigma_A = %.4f', sc/3), ...
        sprintf('3\\sigma_\\phi = %.4f', sp/3), 'Location','best');
else
    pts = s_hat(idx_data);
    plot(real(pts), imag(pts), 'b.', 'MarkerSize', 5); hold on;
    plot(real(const), imag(const), 'ro', 'MarkerSize', 8, 'LineWidth', 1.8);
    legend('Estimados','Constelação ideal','Location','best');
end

axis equal; grid on;
xlabel('I (componente em fase)');
ylabel('Q (componente em quadratura)');
title(sprintf('Elipse de incerteza IQ  |  \\sigma_A=%.4e  \\sigma_\\phi=%.4e', sigma_A, sigma_ph));
save_current_figure(fig_elipse, outDir, ['09_elipse_incerteza_' figTag]);

% Sensibilidade de amplitude vs N2
fig_sensA = figure('Name','Sensibilidade de amplitude vs N2','Position',[200 620 560 380]);
eta_A_vec = zeros(size(N2_vec));
eta_B_vec = zeros(size(N2_vec));
for ki = 1:length(N2_vec)
    Rs_k          = fIF / N2_vec(ki);
    eta_A_vec(ki) = sigma_A * gain_eq / dCfl_dA / sqrt(Rs_k);
    eta_B_vec(ki) = eta_A_vec(ki) * Omega_sig0 * sqrt(2) / gamma_e * 1e9;
end

yyaxis left
semilogy(N2_vec, eta_A_vec, 's-r', 'LineWidth', 1.6, 'MarkerSize', 7);
hold on;
semilogy(N2, eta_A_norm, 'k^', 'MarkerSize', 12, 'LineWidth', 2.2);
ylabel('\eta_A [ampl. norm./√Hz]');

yyaxis right
semilogy(N2_vec, eta_B_vec, 'o--m', 'LineWidth', 1.4, 'MarkerSize', 6);
ylabel('\eta_B [nT/√Hz]');

grid on;
xlabel('N_2 (ciclos IF por símbolo)');
title('Sensibilidade de amplitude do sensor NV vs integração por símbolo');
legend('\eta_A(N_2)', sprintf('Atual N_2=%d', N2), '\eta_B(N_2)', 'Location','northwest');
save_current_figure(fig_sensA, outDir, ['10_sensibilidade_amplitude_' figTag]);

%% =========================================================================
% 20) Diagnóstico espectral
% =========================================================================

n_fft_preview = min(Nsamp_total - Nsamp_cal, 2^20);
F_msg = samp_F(Nsamp_cal + (1:n_fft_preview));
F_msg = F_msg - median(F_msg);
Nfft  = length(F_msg);
w_fft = local_hann(Nfft);
Yfft  = fft(F_msg .* w_fft);
Pfft  = abs(Yfft(1:floor(Nfft/2)+1));
f_ax  = fs_cw*(0:floor(Nfft/2))/Nfft;

[~, ipeak] = max(Pfft(2:end));
f_peak = f_ax(ipeak+1);

fprintf('\n[DIAGNOSTICO ESPECTRAL]\n');
fprintf('  Pico espectral = %.4f MHz\n', f_peak/1e6);
fprintf('  Esperado fIF   = %.4f MHz\n', fIF/1e6);

%% =========================================================================
% 21) Plots
% =========================================================================

sym_times = Tcal + ((0:Nsym_v-1)' + 0.5)*Tsym;
t_data_us = sym_times(idx_data)*1e6;

outDir_qam = outDir;
plot_window = 1:min(Nsamp_total, Nsamp_cal + 4*Nsamp_sym);

apply_ieee_style = @(axObj) set(axObj, 'Box', 'on', 'TickDir', 'in', ...
    'TickLength', [0.015 0.015], 'LineWidth', 1.8, 'FontSize', 22, ...
    'FontWeight', 'bold', 'XMinorTick', 'on', 'YMinorTick', 'on');

% 1) Fluorescência integrada por janela de amostragem
fig1 = figure('Color','w','Position',[80 80 1100 650]); ax = axes('Parent',fig1); hold(ax,'on');
plot(ax, samp_t(plot_window)*1e6, samp_F(plot_window), 'b-', 'LineWidth', 4.6, 'DisplayName', 'Noisy');
if noise_enable
    plot(ax, samp_t(plot_window)*1e6, samp_F_ideal(plot_window), 'Color',[0.55 0.55 0.55], 'LineWidth', 4.6, 'DisplayName', 'Ideal');
end
xline(ax, Tcal*1e6, '--k', 'Calib. end', 'LineWidth', 2.5);
grid(ax,'on'); ax.GridAlpha = 0.12; apply_ieee_style(ax);
xlabel(ax,'Time (\mus)','FontSize',28,'FontWeight','bold'); ylabel(ax,'Integrated F [a.u.]','FontSize',28,'FontWeight','bold');
title(ax,'Integrated fluorescence per sampling window','FontSize',24,'FontWeight','bold');
exportgraphics(fig1, fullfile(outDir_qam, '01_fluorescence.png'), 'Resolution', 600);

% 2) Fluorescência AC
fig2 = figure('Color','w','Position',[80 80 1100 650]); ax = axes('Parent',fig2); hold(ax,'on');
plot(ax, samp_t(plot_window)*1e6, F_ac(plot_window), 'k-', 'LineWidth', 4.6);
xline(ax, Tcal*1e6, '--r', 'Calib. end', 'LineWidth', 2.5);
grid(ax,'on'); ax.GridAlpha = 0.12; apply_ieee_style(ax);
xlabel(ax,'Time (\mus)','FontSize',28,'FontWeight','bold'); ylabel(ax,'F_{AC} [a.u.]','FontSize',28,'FontWeight','bold');
title(ax,'Fluorescence after slow component removal','FontSize',24,'FontWeight','bold');
exportgraphics(fig2, fullfile(outDir_qam, '02_fluorescence_AC.png'), 'Resolution', 600);

% 3) Constelação bruta (antes da correção de fase)
fig3 = figure('Color','w','Position',[80 80 750 700]); ax = axes('Parent',fig3); hold(ax,'on');
plot(ax, real(z_sym(idx_data)), imag(z_sym(idx_data)), 'b.', 'MarkerSize', 24);
axis(ax,'equal'); grid(ax,'on'); ax.GridAlpha = 0.12; apply_ieee_style(ax);
xlabel(ax,'Re\{z_k\}','FontSize',28,'FontWeight','bold'); ylabel(ax,'Im\{z_k\}','FontSize',28,'FontWeight','bold');
title(ax, sprintf('Raw constellation (phi_0 = %.2f rad)', phi0_tx),'FontSize',18,'FontWeight','bold');
exportgraphics(fig3, fullfile(outDir_qam, '03_constellation_raw.png'), 'Resolution', 600);

% 4) Constelação após correção de fase (pré-equalização, normalizada)
fig4 = figure('Color','w','Position',[80 80 750 700]); ax = axes('Parent',fig4); hold(ax,'on');
plot(ax, real(z_sym_corr(idx_data)), imag(z_sym_corr(idx_data)), 'g.', 'MarkerSize', 24);
axis(ax,'equal'); grid(ax,'on'); ax.GridAlpha = 0.12; apply_ieee_style(ax);
xlabel(ax,'Re\{z_{corr}\}','FontSize',28,'FontWeight','bold'); ylabel(ax,'Im\{z_{corr}\}','FontSize',28,'FontWeight','bold');
title(ax, 'After phase/frequency correction (pre-equalization)','FontSize',20,'FontWeight','bold');
exportgraphics(fig4, fullfile(outDir_qam, '04_constellation_post_phase.png'), 'Resolution', 600);

% 5) Constelação equalizada vs ideal
fig5 = figure('Color','w','Position',[80 80 850 750]); ax = axes('Parent',fig5); hold(ax,'on');
plot(ax, real(s_hat_data), imag(s_hat_data), 'b.', 'MarkerSize', 24, 'DisplayName', 'Estimated');
plot(ax, real(const), imag(const), 'ro', 'MarkerSize', 14, 'LineWidth', 3.5, 'DisplayName', 'Ideal constellation');
axis(ax,'equal'); grid(ax,'on'); ax.GridAlpha = 0.12; apply_ieee_style(ax);
xlabel(ax,'I','FontSize',28,'FontWeight','bold'); ylabel(ax,'Q','FontSize',28,'FontWeight','bold');
title(ax, sprintf('%d-QAM without lock-in | phi_0=%.2f rad | EVM=%.2f%% | SER=%.2f%%', ...
    M_qam, phi0_tx, EVM, SER),'FontSize',18,'FontWeight','bold');
lim = 1.35*max(abs([real(const); imag(const); real(s_hat_data); imag(s_hat_data)]));
xlim(ax, [-lim lim]); ylim(ax, [-lim lim]);
lgd = legend(ax,'Location','northwest'); set(lgd,'FontSize',18,'FontWeight','bold','Box','on');
exportgraphics(fig5, fullfile(outDir_qam, '05_constellation_equalized.png'), 'Resolution', 600);

% 6) Componentes I e Q no tempo
fig6 = figure('Color','w','Position',[80 80 1100 750]);

ax_i = subplot(2,1,1,'Parent',fig6); hold(ax_i,'on');
stairs(ax_i, t_data_us, real(s_tx_data), 'b', 'LineWidth', 4.6, 'DisplayName', 'Tx');
stairs(ax_i, t_data_us, real(s_dec(idx_data)), 'r--', 'LineWidth', 3.5, 'DisplayName', 'Decided');
grid(ax_i,'on'); ax_i.GridAlpha = 0.12; apply_ieee_style(ax_i);
ylabel(ax_i, 'I', 'FontSize', 28, 'FontWeight', 'bold');
title(ax_i, 'Component I', 'FontSize', 22, 'FontWeight', 'bold');
lgd_i = legend(ax_i, 'Location', 'best'); set(lgd_i, 'FontSize', 16, 'Box', 'on');

ax_q = subplot(2,1,2,'Parent',fig6); hold(ax_q,'on');
stairs(ax_q, t_data_us, imag(s_tx_data), 'b', 'LineWidth', 4.6, 'DisplayName', 'Tx');
stairs(ax_q, t_data_us, imag(s_dec(idx_data)), 'r--', 'LineWidth', 3.5, 'DisplayName', 'Decided');
grid(ax_q,'on'); ax_q.GridAlpha = 0.12; apply_ieee_style(ax_q);
xlabel(ax_q, 'Time (\mus)', 'FontSize', 28, 'FontWeight', 'bold'); ylabel(ax_q, 'Q', 'FontSize', 28, 'FontWeight', 'bold');
title(ax_q, 'Component Q', 'FontSize', 22, 'FontWeight', 'bold');
lgd_q = legend(ax_q, 'Location', 'best'); set(lgd_q, 'FontSize', 16, 'Box', 'on');

exportgraphics(fig6, fullfile(outDir_qam, '06_components_IQ.png'), 'Resolution', 600);

% 7) Distribuição do erro residual
fig7 = figure('Color','w','Position',[80 80 750 700]); ax = axes('Parent',fig7); hold(ax,'on');
plot(ax, real(noise_resid), imag(noise_resid), 'k.', 'MarkerSize', 24);
axis(ax,'equal'); grid(ax,'on'); ax.GridAlpha = 0.12; apply_ieee_style(ax);
xlabel(ax,'Error in I','FontSize',28,'FontWeight','bold'); ylabel(ax,'Error in Q','FontSize',28,'FontWeight','bold');
title(ax, sprintf('Residual error | std_I=%.3f | std_Q=%.3f', ...
    std(real(noise_resid)), std(imag(noise_resid))),'FontSize',20,'FontWeight','bold');
exportgraphics(fig7, fullfile(outDir_qam, '07_residual_error.png'), 'Resolution', 600);

% 8) Espectro de magnitude da fluorescência
fig8 = figure('Color','w','Position',[80 80 1100 650]); ax = axes('Parent',fig8); hold(ax,'on');
plot(ax, f_ax/1e6, 20*log10(Pfft/max(Pfft) + eps), '-', 'LineWidth', 4.6);
xline(ax, fIF/1e6, '--r', 'f_{IF}', 'LineWidth', 2.5);
grid(ax,'on'); ax.GridAlpha = 0.12; apply_ieee_style(ax);
xlabel(ax,'Frequency [MHz]','FontSize',28,'FontWeight','bold'); ylabel(ax,'Normalized magnitude [dB]','FontSize',28,'FontWeight','bold');
title(ax,'Fluorescence spectrum after calibration','FontSize',24,'FontWeight','bold');
xlim(ax,[0 3*fIF/1e6]);
exportgraphics(fig8, fullfile(outDir_qam, '08_spectrum.png'), 'Resolution', 600);

%% -------------------------
% Salvar dados para o Python
%% -------------------------
out_QAM.samp_t      = samp_t;
out_QAM.samp_F      = samp_F;
out_QAM.F_ac        = F_ac;
out_QAM.F_slow      = F_slow;
out_QAM.Tsym        = Tsym;
out_QAM.Nsym        = Nsym;
out_QAM.fIF         = fIF;
out_QAM.fs_cw       = fs_cw;
out_QAM.Ntrain_sym  = Ntrain_sym;
out_QAM.Ndata_sym   = Ndata_sym;
out_QAM.s_sym       = s_sym;
out_QAM.z_sym       = z_sym;
out_QAM.z_scale     = z_scale;
out_QAM.s_hat       = s_hat;
out_QAM.a_eq        = a_eq;
out_QAM.b_eq        = b_eq;
out_QAM.EVM         = EVM;
out_QAM.SER         = SER;
out_QAM.nv_rates = nv_rates;
out_QAM.R532_cw = R532_cw;
out_QAM.Omega_sig0 = Omega_sig0;
out_QAM.Omega_ref0 = Omega_ref0;
out_QAM.mean_ideal_detector_counts = N_NV_eff*eta_det*mean(samp_F_ideal);
out_QAM.f_peak_diagnostic = f_peak;
save(fullfile(outDir, 'out_QAM.mat'), 'out_QAM');
fprintf('\n[CONCLUIDO] Script finalizado.\n');

%% =========================================================================
% Funções locais
% =========================================================================

function L = lindblad_superop(A, d)
    I   = eye(d);
    AdA = A'*A;
    L   = kron(conj(A), A) ...
        - 0.5*kron(I, AdA) ...
        - 0.5*kron(AdA.', I);
end

function save_current_figure(hFig, outDir, baseName)
    pngPath = fullfile(outDir, [baseName '.png']);
    pdfPath = fullfile(outDir, [baseName '.pdf']);
    figPath = fullfile(outDir, [baseName '.fig']);
    try
        exportgraphics(hFig, pngPath, 'Resolution', 300);
        exportgraphics(hFig, pdfPath, 'ContentType', 'vector');
    catch
        saveas(hFig, pngPath);
        saveas(hFig, pdfPath);
    end
    savefig(hFig, figPath);
    fprintf('  Figura salva: %s\n', pngPath);
end

function w = local_hann(N)
    if N <= 1
        w = ones(N,1);
    else
        n = (0:N-1)';
        w = 0.5 - 0.5*cos(2*pi*n/(N-1));
    end
end