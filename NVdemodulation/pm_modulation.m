% Execute este script para salvar os resultados ao lado deste arquivo.
clear; clc; close all;
script_dir = fileparts(mfilename('fullpath'));

%% =========================================================================
% simular_pm.m
%
% CASO COM LASER PULSADO
%
% Versao com:
%   1. Evolucao no laboratorio, como no codigo que deu RMSE menor
%   2. dt = 20 ps para resolver as portadoras GHz
%      (relaxacao u -> e com tau_ue = 100 fs integrada exatamente por
%       splitting de Strang; o RK4 so ve a parte nao rigida)
%   3. Laser pulsado
%   4. Fluorescencia integrada pulso a pulso
%   5. Lock-in pulso a pulso
%   6. Filtro Gaussiano com filtfilt
%   7. Sem extrapolacao nas bordas
%   8. Remocao de bordas/transientes no RMSE
%   9. Correcao por Hilbert/quadratura:
%
%          m_true ~= a*m_est + b*Hilbert(m_est)
%
%      Isso corrige ganho e fase dinamica na frequencia da mensagem.
%
% Base:
%  1..3 : |g,+1>, |g,0>, |g,-1>
%  4..6 : |e,+1>, |e,0>, |e,-1>
%  7..9 : |u,+1>, |u,0>, |u,-1>
%  10   : |s>
% =========================================================================

%% -------------------------------------------------------------------------
% 1. Inputs fisicos
% -------------------------------------------------------------------------

Bstatic = [0 0 5e-3];       % T

% Tempo total
Tcal = 2e-6;                % s, intervalo inicial sem PM para calibrar fase

% Janela com 16 bits NRZ de 1 us.
Tmsg = 16e-6;               % s
Ttot = Tcal + Tmsg;         % s

% Direcao da MW
mw_dir = [1 0 0];
mw_dir = mw_dir/norm(mw_dir); o 

% Fonte SIG
f_sig   = 3.000e9;          % Hz
S_sig   = 1e8;              % W/m^2
phi_sig = 0;                % rad

% Fonte REF
% fIF = 11 MHz, abaixo de Nyquist para frep = 50 MHz.
f_ref   = 3.011e9;          % Hz
S_ref   = 1e8;              % W/m^2
phi_ref = 0;                % rad

% Frequencia intermediaria
fIF = abs(f_ref - f_sig);

% Laser 532 nm pulsado
P532  = 40e-3;              % W
k_532 = 1e10;                % (1/s)/W
R_532 = k_532*P532;         % 1/s durante o pulso

Tpulse   = 20e-9;           % periodo dos pulsos opticos
t_laser  = 5e-9;            % duracao do pump optico ON
t_detect = 15e-9;           % janela de integracao de fluorescencia

% Exclui o transiente inicial de reenchimento optico.
t_skip = 2e-9;              % s

frep = 1/Tpulse;

if frep <= 2*fIF
    error('A taxa de repeticao dos pulsos deve ser maior que 2*fIF.');
end

if t_skip >= t_detect
    error('t_skip deve ser menor que t_detect.');
end

% Passo temporal no laboratorio
dt = 2e-11;                 % 20 ps
Nt = floor(Ttot/dt) + 1;
t = (0:Nt-1)'*dt;
gate_steps = round([Tpulse,t_laser,t_skip,t_detect]/dt);
assert(all(abs(gate_steps*dt - [Tpulse,t_laser,t_skip,t_detect]) < 1e-8*dt), ...
    'As bordas das janelas devem coincidir com a malha de integração.');
phase_steps = mod((0:Nt-1)',gate_steps(1));
laser_on_vec = phase_steps < gate_steps(2);
detect_on_vec = phase_steps >= gate_steps(3) & phase_steps < gate_steps(4);

fprintf('\n=== CONFIGURACAO HETERODINA COM LASER PULSADO ===\n');
fprintf('f_sig  = %.6f GHz\n', f_sig/1e9);
fprintf('f_ref  = %.6f GHz\n', f_ref/1e9);
fprintf('fIF    = %.3f MHz\n', fIF/1e6);
fprintf('frep   = %.3f MHz\n', frep/1e6);
fprintf('Tpulse = %.3f ns\n', Tpulse*1e9);
fprintf('tlaser = %.3f ns\n', t_laser*1e9);
fprintf('tdet   = %.3f ns\n', t_detect*1e9);
fprintf('tskip  = %.3f ns\n', t_skip*1e9);
fprintf('Tcal   = %.3f us\n', Tcal*1e6);
fprintf('Tmsg   = %.3f us\n', Tmsg*1e6);
fprintf('Ttot   = %.3f us\n', Ttot*1e6);
fprintf('dt     = %.3f ps\n', dt*1e12);
fprintf('Nt     = %d\n', Nt);

%% -------------------------------------------------------------------------
% 2. Mensagem PM
% -------------------------------------------------------------------------

% fm = 0.25e6;                % Hz
% 
% % Para testar linearidade, voce pode usar kp = 0.3.
% % Para comparar com seu codigo anterior, use kp = 0.9.
% % Para compromisso, use kp = 0.7.
% kp = 0.7;                   % rad
% 
% m = zeros(Nt,1);
% 
% idx_msg = t >= Tcal;
% 
% % Mensagem senoidal
% m(idx_msg) = sin(2*pi*fm*(t(idx_msg)-Tcal));
% 
% % Para testar mensagem quadrada depois:
% % tau_msg = t(idx_msg) - Tcal;
% % m(idx_msg) = 2*(mod(fm*tau_msg,1) < 0.5) - 1;
% 
% beta = kp*m;
% 
% fprintf('\n=== MENSAGEM PM ===\n');
% fprintf('fm = %.3f MHz\n', fm/1e6);
% fprintf('kp = %.3f rad\n', kp);
% fprintf('J1(kp)/kp = %.6f\n', besselj(1,kp)/kp);

%% -------------------------------------------------------------------------
% 2. Mensagem digital PM — bits NRZ
% -------------------------------------------------------------------------

% Taxa de bits
Rb   = 1.0e6;               % bits/s
Tbit = 1/Rb;                % duracao de cada bit

% Numero de bits dentro da janela de mensagem
Nbits = floor(Tmsg/Tbit);

% Ajusta Tmsg efetivo para conter numero inteiro de bits
Tmsg_eff = Nbits*Tbit;

if abs(Tmsg_eff - Tmsg) > 1e-15
    fprintf('\n[INFO] Tmsg ajustado de %.3f us para %.3f us para conter bits inteiros.\n', ...
        Tmsg*1e6, Tmsg_eff*1e6);
end

% Sequencia de bits
% Use rng fixo para reprodutibilidade
rng(7);
bits_tx = randi([0 1],Nbits,1);

% Mapeamento bipolar:
% bit 0 -> -1
% bit 1 -> +1
symbols_tx = 2*bits_tx - 1;

% Monta m(t)
m = zeros(Nt,1);

idx_msg = t >= Tcal & t < Tcal + Tmsg_eff;

tau_msg = t(idx_msg) - Tcal;

bit_index = floor(tau_msg/Tbit) + 1;
bit_index(bit_index < 1) = 1;
bit_index(bit_index > Nbits) = Nbits;

m(idx_msg) = symbols_tx(bit_index);

% Fora da janela de mensagem, m = 0.
kp = 0.7;                   % rad
beta = kp*m;

fprintf('\n=== MENSAGEM DIGITAL PM ===\n');
fprintf('Rb        = %.3f Mbps\n', Rb/1e6);
fprintf('Tbit      = %.3f us\n', Tbit*1e6);
fprintf('Nbits     = %d\n', Nbits);
fprintf('kp        = %.3f rad\n', kp);
fprintf('J1(kp)/kp = %.6f\n', besselj(1,kp)/kp);
fprintf('bits_tx   = ');
fprintf('%d',bits_tx);
fprintf('\n');

%% -------------------------------------------------------------------------
% 3. Taxas dissipativas
% -------------------------------------------------------------------------

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

%% -------------------------------------------------------------------------
% 4. Conversao MW: S -> B
% -------------------------------------------------------------------------

mu0 = 4*pi*1e-7;
c0  = 299792458;

Bsig_amp = sqrt(mu0*S_sig/c0);
Bref_amp = sqrt(mu0*S_ref/c0);

fprintf('\n=== CAMPOS MW ===\n');
fprintf('Bsig = %.6e T\n', Bsig_amp);
fprintf('Bref = %.6e T\n', Bref_amp);

%% -------------------------------------------------------------------------
% 5. Operadores de spin e Hamiltonianos estaticos
% -------------------------------------------------------------------------

gamma_e = 2*pi*28e9;         % rad/s/T

Dg = 2*pi*2.87e9;            % rad/s
De = 2*pi*1.42e9;            % rad/s
Du = 2*pi*1.42e9;            % rad/s

Es = 0;

Sx = (1/sqrt(2))*[ ...
    0 1 0;
    1 0 1;
    0 1 0];

Sy = (1/sqrt(2))*[ ...
    0 -1i 0;
    1i 0 -1i;
    0 1i 0];

Sz = [ ...
    1 0 0;
    0 0 0;
    0 0 -1];

Hspin = @(D,Bvec) D*(Sz*Sz) ...
                + gamma_e*(Bvec(1)*Sx + Bvec(2)*Sy + Bvec(3)*Sz);

Hg = Hspin(Dg,Bstatic);
He = Hspin(De,Bstatic);
Hu = Hspin(Du,Bstatic);

d   = 10;
I10 = eye(d);

Hstatic = zeros(d);
Hstatic(1:3,1:3) = Hg;
Hstatic(4:6,4:6) = He;
Hstatic(7:9,7:9) = Hu;
Hstatic(10,10)   = Es;

f_plus  = Dg/(2*pi) + gamma_e*Bstatic(3)/(2*pi);
f_minus = Dg/(2*pi) - gamma_e*Bstatic(3)/(2*pi);

fprintf('\n=== TRANSICOES GROUND ===\n');
fprintf('f_plus  = %.6f GHz\n', f_plus/1e9);
fprintf('f_minus = %.6f GHz\n', f_minus/1e9);
fprintf('detuning SIG em |+1> = %.3f MHz\n', (f_plus-f_sig)/1e6);
fprintf('detuning REF em |+1> = %.3f MHz\n', (f_plus-f_ref)/1e6);

%% -------------------------------------------------------------------------
% 6. Indices dos estados
% -------------------------------------------------------------------------

igp = 1;  ig0 = 2;  igm = 3;
iep = 4;  ie0 = 5;  iem = 6;
iup = 7;  iu0 = 8;  ium = 9;
is  = 10;

%% -------------------------------------------------------------------------
% 7. Operadores de Lindblad sem pump optico
% -------------------------------------------------------------------------

L = {};
Lue_ops = {};

% u -> e spin-conserving (taxa 1e13 /s: parte rigida, ver secao 9)
for mm = 1:3
    iu = 6 + mm;
    ie = 3 + mm;

    A = zeros(d);
    A(ie,iu) = sqrt(Gamma_ue);
    Lue_ops{end+1} = A;     % separado: tratado por propagador exato (rigido)
end

% Decaimento radiativo e -> g
A = zeros(d); A(igp,iep) = sqrt(Gamma_rpm); L{end+1} = A;
A = zeros(d); A(ig0,ie0) = sqrt(Gamma_r0 ); L{end+1} = A;
A = zeros(d); A(igm,iem) = sqrt(Gamma_rpm); L{end+1} = A;

% ISC e -> s
A = zeros(d); A(is,iep) = sqrt(Gamma_isc_pm); L{end+1} = A;
A = zeros(d); A(is,ie0) = sqrt(Gamma_isc0 ); L{end+1} = A;
A = zeros(d); A(is,iem) = sqrt(Gamma_isc_pm); L{end+1} = A;

% s -> g
A = zeros(d); A(igp,is) = sqrt(Gamma_sgpm_eq); L{end+1} = A;
A = zeros(d); A(ig0,is) = sqrt(Gamma_sg0_eq ); L{end+1} = A;
A = zeros(d); A(igm,is) = sqrt(Gamma_sgpm_eq); L{end+1} = A;

%% -------------------------------------------------------------------------
% 8. Pump optico 532 nm g -> u
% -------------------------------------------------------------------------

Lp = {};

for mm = 1:3
    ig = mm;
    iu = 6 + mm;

    A = zeros(d);
    A(iu,ig) = sqrt(R_532);
    Lp{end+1} = A;
end

%% -------------------------------------------------------------------------
% 9. Pre-calculo dos Liouvillianos fixos
% -------------------------------------------------------------------------

% Dissipacao sem pump optico
Ldiss_no_pump = zeros(d*d,d*d);

for n = 1:numel(L)
    A = L{n};
    K = A'*A;

    Ldiss_no_pump = Ldiss_no_pump ...
        + kron(conj(A),A) ...
        - 0.5*kron(I10,K) ...
        - 0.5*kron(K.',I10);
end

% Pump optico, ligado/desligado por laser_gate(t)
Lpump = zeros(d*d,d*d);

for n = 1:numel(Lp)
    A = Lp{n};
    K = A'*A;

    Lpump = Lpump ...
        + kron(conj(A),A) ...
        - 0.5*kron(I10,K) ...
        - 0.5*kron(K.',I10);
end

% Parte coerente fixa
Lcoh_static = -1i*(kron(I10,Hstatic) - kron(Hstatic.',I10));

% Liouvilliano base sem pump
Lbase = sparse(Ldiss_no_pump + Lcoh_static);
Lpump = sparse(Lpump);

% -------------------------------------------------------------------------
% Parte rigida: relaxacao vibronica u -> e (tau_ue = 100 fs)
%
% Com dt = 20 ps, Gamma_ue*dt = 200, muito acima do limite de estabilidade
% do RK4 (|lambda*dt| ~ 2.8). Por isso o termo u -> e e separado e
% integrado EXATAMENTE por splitting de Strang:
%
%   y <- expm(Lue*dt/2) * y
%   y <- RK4( Lbase + gate*Lpump + Lmw , dt ) * y
%   y <- expm(Lue*dt/2) * y
%
% O restante do Liouvilliano tem |lambda|max*dt ~ 0.4 (portadoras GHz),
% dentro da regiao estavel do RK4. Os niveis |u> continuam no modelo.
% -------------------------------------------------------------------------
Lue = zeros(d*d,d*d);
for n = 1:numel(Lue_ops)
    A = Lue_ops{n};
    K = A'*A;
    Lue = Lue + kron(conj(A),A) - 0.5*kron(I10,K) - 0.5*kron(K.',I10);
end

E_ue_half = sparse(expm(Lue*dt/2));     % propagador exato de meio passo

% Geradores MW pre-calculados: Lmw = Bx*GX + By*GY + Bz*GZ
blk3 = @(S3) blkdiag(S3, zeros(7));          % micro-ondas no tripleto fundamental
mw_super = @(H) -1i*(kron(I10,H) - kron(H.',I10));
GX = sparse(mw_super(gamma_e*blk3(Sx)));
GY = sparse(mw_super(gamma_e*blk3(Sy)));
GZ = sparse(mw_super(gamma_e*blk3(Sz)));

lam_rk4 = max(abs(eig(full(Lbase + Lpump))));
fprintf('\n=== ESTABILIDADE ===\n');
fprintf('Gamma_ue*dt (tratado exato)       = %.3g\n', Gamma_ue*dt);
fprintf('|lambda|max*dt (parte em RK4)     = %.3g  (limite ~2.8)\n', lam_rk4*dt);
if lam_rk4*dt > 2.5
    error('dt grande demais para a parte nao rigida: reduza dt.');
end

%% -------------------------------------------------------------------------
% 10. Estado inicial e buffers
% -------------------------------------------------------------------------

rho0 = zeros(d);
rho0(ig0,ig0) = 1;
y = rho0(:);

Pg_p = zeros(Nt,1);
Pg_0 = zeros(Nt,1);
Pg_m = zeros(Nt,1);

Pe_p = zeros(Nt,1);
Pe_0 = zeros(Nt,1);
Pe_m = zeros(Nt,1);

Pu_p = zeros(Nt,1);
Pu_0 = zeros(Nt,1);
Pu_m = zeros(Nt,1);

Ps    = zeros(Nt,1);
Fluor = zeros(Nt,1);
Trrho = zeros(Nt,1);

rho01 = zeros(Nt,1);
phi01 = zeros(Nt,1);


%% -------------------------------------------------------------------------
% 11. Loop temporal com laser pulsado
% -------------------------------------------------------------------------

fprintf('\n=== SIMULANDO DINAMICA COM LASER PULSADO ===\n');

rho01_ref_defined = false;
rho01_ref = 0;

for k = 1:Nt

    rho = reshape(y,d,d);

    Pg_p(k) = real(rho(igp,igp));
    Pg_0(k) = real(rho(ig0,ig0));
    Pg_m(k) = real(rho(igm,igm));

    Pe_p(k) = real(rho(iep,iep));
    Pe_0(k) = real(rho(ie0,ie0));
    Pe_m(k) = real(rho(iem,iem));

    Pu_p(k) = real(rho(iup,iup));
    Pu_0(k) = real(rho(iu0,iu0));
    Pu_m(k) = real(rho(ium,ium));

    Ps(k) = real(rho(is,is));

    Trrho(k) = real(trace(rho));

    % Fluorescencia radiativa instantanea
    Fluor(k) = Gamma_rpm*real(rho(iep,iep)) ...
             + Gamma_r0 *real(rho(ie0,ie0)) ...
             + Gamma_rpm*real(rho(iem,iem));

    % Coerencia
    rho01(k) = rho(ig0,igp);

    if ~rho01_ref_defined && abs(rho01(k)) > 1e-12 && t(k) > 0.5e-6
        rho01_ref = rho01(k);
        rho01_ref_defined = true;
    end

    if rho01_ref_defined
        phi01(k) = angle(rho01(k)) - angle(rho01_ref);
    else
        phi01(k) = 0;
    end

    if k == Nt
        break;
    end

    tk = t(k);

    laser_gate = double(laser_on_vec(k));

    % Micro-ondas heterodinas no laboratorio
    beta_k = beta(k);

    Bsig_t = (Bsig_amp*cos(2*pi*f_sig*tk + phi_sig + beta_k))*mw_dir;
    Bref_t = (Bref_amp*cos(2*pi*f_ref*tk + phi_ref))*mw_dir;

    Bmw_t = Bsig_t + Bref_t;

    Bx = Bmw_t(1);
    By = Bmw_t(2);
    Bz = Bmw_t(3);

    Lmw = Bx*GX + By*GY + Bz*GZ;

    Lt = Lbase + laser_gate*Lpump + Lmw;

    % Strang: meio passo exato u->e, RK4 no resto, meio passo exato u->e
    y = E_ue_half*y;

    k1 = Lt*y;
    k2 = Lt*(y + 0.5*dt*k1);
    k3 = Lt*(y + 0.5*dt*k2);
    k4 = Lt*(y + dt*k3);

    y = y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);

    y = E_ue_half*y;

    % Correcao leve de traco e hermiticidade
    if mod(k,200000) == 0
        fprintf('  %8d/%8d  (%.0f%%)\n', k, Nt, 100*k/Nt);
    end

    if mod(k,2000) == 0
        rho_tmp = reshape(y,d,d);
        rho_tmp = 0.5*(rho_tmp + rho_tmp');

        tr_tmp = trace(rho_tmp);
        if abs(tr_tmp) > 0
            rho_tmp = rho_tmp/tr_tmp;
        end

        rho_diag = real(diag(rho_tmp));

        if any(~isfinite(rho_diag)) || max(abs(rho_diag)) > 10
            error('Instabilidade numerica detectada em t = %.3f us', tk*1e6);
        end

        y = rho_tmp(:);
    end
end

% Preenche ultimo valor de gating

phi01 = unwrap(phi01);
phi01 = phi01 - phi01(1);

fprintf('\n=== INFORMACOES GERAIS ===\n');
fprintf('f_sig = %.6f GHz\n', f_sig/1e9);
fprintf('f_ref = %.6f GHz\n', f_ref/1e9);
fprintf('fIF   = %.3f MHz\n', fIF/1e6);
fprintf('Bsig  = %.3e T\n', Bsig_amp);
fprintf('Bref  = %.3e T\n', Bref_amp);
fprintf('Nt    = %d\n', Nt);
fprintf('trace final(rho) = %.12f\n', Trrho(end));
fprintf('Fluor min = %.3e | max = %.3e\n', min(Fluor), max(Fluor));

%% -------------------------------------------------------------------------
% 12. Integracao da fluorescencia pulso a pulso
% -------------------------------------------------------------------------

Npulses = floor((Nt-1)/gate_steps(1));

pulse_t    = zeros(Npulses,1);
pulse_F    = zeros(Npulses,1);
for n = 1:Npulses
    start_step = (n-1)*gate_steps(1);
    idx_p = start_step+(gate_steps(3)+1:gate_steps(4));
    pulse_t(n) = ((n-1)*Tpulse) + 0.5*(t_skip+t_detect);
    pulse_F(n) = sum(Fluor(idx_p))*dt;
end
pulse_m = interp1(t,m,pulse_t,'previous');
pulse_beta = interp1(t,beta,pulse_t,'previous');

valid_pulse = isfinite(pulse_t) & isfinite(pulse_F) & pulse_t > 0;

pulse_t    = pulse_t(valid_pulse);
pulse_F    = pulse_F(valid_pulse);
pulse_m    = pulse_m(valid_pulse);
pulse_beta = pulse_beta(valid_pulse);

%% -------------------------------------------------------------------------
% 13. Lock-in pulso a pulso em fIF
% -------------------------------------------------------------------------

t_trans = 0.8e-6;

idx_cal_p = (pulse_t >= t_trans) & (pulse_t < Tcal);
idx_rec_p = pulse_t >= Tcal;

if sum(idx_cal_p) < 10
    error('Janela de calibracao pequena demais. Aumente Tcal ou reduza t_trans.');
end

% Remove DC usando apenas janela de calibracao sem PM apos transiente
F_mean_p = mean(pulse_F(idx_cal_p));
F_ac_p = pulse_F - F_mean_p;

% Demodulacao complexa em fIF
z_raw_p = F_ac_p .* exp(-1i*2*pi*fIF*pulse_t);

% Filtro passa-baixa Gaussiano com filtfilt
dt_p = Tpulse;

% BW_lpf  = 5*fm;
% Para bits NRZ, a banda precisa passar transicoes de bit,
% mas ainda rejeitar bem o batimento em fIF.
BW_lpf = min(1.5*Rb, 0.25*fIF);
sigma_t = 1/(2*pi*BW_lpf);

N_half = ceil(3*sigma_t/dt_p);
t_kern = (-N_half:N_half)'*dt_p;

kern_p = exp(-0.5*(t_kern/sigma_t).^2);
kern_p = kern_p/sum(kern_p);

z_lock_p = local_zero_phase_fir(kern_p, z_raw_p);

X_lock_p = real(z_lock_p);
Y_lock_p = imag(z_lock_p);

theta_lock_p = unwrap(angle(z_lock_p));

% Fase de calibracao quando beta = 0
theta0_p = angle(mean(exp(1i*theta_lock_p(idx_cal_p))));

theta_PL_p = unwrap(theta_lock_p - theta0_p);
theta_PL_p = theta_PL_p - mean(theta_PL_p(idx_cal_p));

fprintf('\n=== LOCK-IN PULSO A PULSO ===\n');
fprintf('BW_lpf = %.3f MHz\n', BW_lpf/1e6);
fprintf('Kernel = %d pulsos = %.3f ns\n', numel(kern_p), numel(kern_p)*Tpulse*1e9);

%% -------------------------------------------------------------------------
% 14. Recuperacao inicial de beta(t)
% -------------------------------------------------------------------------

% Estimativa inicial pela fase do lock-in.
beta_rec_p = -theta_PL_p;

% Remove offset pela janela sem PM
beta_rec_p = beta_rec_p - mean(beta_rec_p(idx_cal_p));

%% -------------------------------------------------------------------------
% Detector de bits
% -------------------------------------------------------------------------
%
% Aqui medimos erro de bits, nao RMSE analogico.
% O receptor amostra a mensagem recuperada no centro de cada bit.
%
% Como o sistema pode ter atraso dinamico, fazemos uma busca pequena de
% atraso de amostragem usando um preambulo de treinamento.

m_rec_p = beta_rec_p/kp;

% Tempos nominais dos centros dos bits
bit_center_t = Tcal + ((0:Nbits-1)' + 0.5)*Tbit;

% Numero de bits de treinamento para escolher atraso, sinal e limiar
Ntrain = min(8,max(4,floor(0.25*Nbits)));

train_idx = 1:Ntrain;
test_idx  = (Ntrain+1):Nbits;

if isempty(test_idx)
    error('Poucos bits para teste. Aumente Tmsg ou Rb.');
end

% Busca de atraso de amostragem
% Procuramos dentro de +-0.45 bit.
tau_grid = linspace(-0.45*Tbit,0.45*Tbit,401);

best_metric = Inf;
best_tau = 0;
best_coef = [1;0];

symbols_train = symbols_tx(train_idx);

for q = 1:numel(tau_grid)

    tau_q = tau_grid(q);

    y_train = interp1( ...
        pulse_t, ...
        m_rec_p, ...
        bit_center_t(train_idx) + tau_q, ...
        'linear', ...
        NaN);

    valid_train = isfinite(y_train);

    if sum(valid_train) < 3
        continue;
    end

    yv = y_train(valid_train);
    sv = symbols_train(valid_train);

    % Ajuste linear:
    % symbol ~= a*y + b
    A = [yv, ones(size(yv))];
    coef_q = A\sv;

    score_train = A*coef_q;
    bits_train_hat = score_train > 0;
    bits_train_true = bits_tx(train_idx(valid_train));

    ber_train_q = mean(bits_train_hat ~= bits_train_true);

    % Criterio secundario: MSE do score contra simbolos verdadeiros
    mse_train_q = mean((sv - score_train).^2);

    metric_q = ber_train_q + 1e-3*mse_train_q;

    if metric_q < best_metric
        best_metric = metric_q;
        best_tau = tau_q;
        best_coef = coef_q;
    end
end

fprintf('\n=== SINCRONISMO DE BITS ===\n');
fprintf('Ntrain       = %d bits\n',Ntrain);
fprintf('tau_amostra  = %.3f ns\n',best_tau*1e9);
fprintf('coef a       = %.6e\n',best_coef(1));
fprintf('coef b       = %.6e\n',best_coef(2));

% Amostra todos os bits com o melhor atraso encontrado
y_bits = interp1( ...
    pulse_t, ...
    m_rec_p, ...
    bit_center_t + best_tau, ...
    'linear', ...
    NaN);

valid_bits = isfinite(y_bits);

score_bits = best_coef(1)*y_bits + best_coef(2);

bits_rx = score_bits > 0;

% Calcula BER no conjunto todo, treinamento e teste
valid_all_idx = find(valid_bits);

bit_errors_all = bits_rx(valid_bits) ~= bits_tx(valid_bits);
BER_all = mean(bit_errors_all);

valid_train_mask = false(Nbits,1);
valid_train_mask(train_idx) = true;
valid_train_mask = valid_train_mask & valid_bits;

valid_test_mask = false(Nbits,1);
valid_test_mask(test_idx) = true;
valid_test_mask = valid_test_mask & valid_bits;

BER_train = mean(bits_rx(valid_train_mask) ~= bits_tx(valid_train_mask));
BER_test  = mean(bits_rx(valid_test_mask)  ~= bits_tx(valid_test_mask));

Nerr_all   = sum(bits_rx(valid_bits) ~= bits_tx(valid_bits));
Nvalid_all = sum(valid_bits);

Nerr_test   = sum(bits_rx(valid_test_mask) ~= bits_tx(valid_test_mask));
Nvalid_test = sum(valid_test_mask);

fprintf('\n=== RESULTADO DE BITS ===\n');
fprintf('BER total = %.6e  (%d erros em %d bits validos)\n', ...
    BER_all,Nerr_all,Nvalid_all);
fprintf('BER treino = %.6e\n',BER_train);
fprintf('BER teste  = %.6e  (%d erros em %d bits de teste)\n', ...
    BER_test,Nerr_test,Nvalid_test);

fprintf('bits_tx = ');
fprintf('%d',bits_tx.');
fprintf('\n');

fprintf('bits_rx = ');
fprintf('%d',bits_rx.');
fprintf('\n');

% Correcao de convencao global de sinal por correlacao
beta_true_p = pulse_beta(idx_rec_p);
beta_est_p  = beta_rec_p(idx_rec_p);

beta_true0_p = beta_true_p - mean(beta_true_p);
beta_est0_p  = beta_est_p  - mean(beta_est_p);

corr_val = sum(beta_true0_p .* beta_est0_p);

if corr_val < 0
    beta_rec_p = -beta_rec_p;
    beta_est_p = -beta_est_p;
    beta_est0_p = -beta_est0_p;
    fprintf('\n[INFO] Sinal global invertido por correlacao cruzada.\n');
end

%% -------------------------------------------------------------------------
% 15. Correcao por Hilbert/quadratura e remocao de bordas do RMSE
% -------------------------------------------------------------------------
%
% O sistema NV + laser pulsado nao responde como ganho real instantaneo.
% Ele introduz fase dinamica. Para mensagem senoidal, a correcao mais limpa e:
%
%   m_true(t) ~= a*m_est(t) + b*H[m_est](t)
%
% onde H[] e a quadratura de Hilbert.
%
% Isso equivale a ajustar ganho complexo na frequencia da mensagem.

m_rec_p_raw = beta_rec_p/kp;

m_true_p = pulse_m(idx_rec_p);
m_est_p  = m_rec_p_raw(idx_rec_p);
t_rec_p  = pulse_t(idx_rec_p);

m_true0_p = m_true_p - mean(m_true_p);
m_est0_p  = m_est_p  - mean(m_est_p);

beta_true_p = pulse_beta(idx_rec_p);
beta_est_p  = beta_rec_p(idx_rec_p);

beta_true0_p = beta_true_p - mean(beta_true_p);
beta_est0_p  = beta_est_p  - mean(beta_est_p);

% Remove bordas/transientes:
%   - borda do filtro filtfilt
%   - inicio da janela de mensagem
%   - fim da janela, onde ajustes ficam sensiveis
guard_time = max([0.5e-6, 4*sigma_t, 4*Tpulse]);

valid_rmse = (t_rec_p > min(t_rec_p) + guard_time) ...
           & (t_rec_p < max(t_rec_p) - guard_time);

if sum(valid_rmse) < 20
    error('Poucos pontos validos para RMSE apos remocao de bordas. Reduza guard_time ou aumente Tmsg.');
end

t_eval = t_rec_p(valid_rmse);

m_true_eval = m_true0_p(valid_rmse);
m_est_eval  = m_est0_p(valid_rmse);

m_true_eval = m_true_eval - mean(m_true_eval);
m_est_eval  = m_est_eval  - mean(m_est_eval);

% Quadratura de Hilbert da estimativa
m_est_quad = local_quadrature(m_est_eval);

% Regressao linear: true ~= a*est + b*quad
Afit = [m_est_eval, m_est_quad];

coef_m = Afit \ m_true_eval;

m_est_corr = Afit*coef_m;

rmse_m = sqrt(mean((m_true_eval - m_est_corr).^2));
rmse_norm = rmse_m/(max(m_true_eval)-min(m_true_eval)+eps);

gain_m  = sqrt(coef_m(1)^2 + coef_m(2)^2);
phase_m = atan2(coef_m(2),coef_m(1));

% Mesma ideia para beta
beta_true_eval = beta_true0_p(valid_rmse);
beta_est_eval  = beta_est0_p(valid_rmse);

beta_true_eval = beta_true_eval - mean(beta_true_eval);
beta_est_eval  = beta_est_eval  - mean(beta_est_eval);

beta_est_quad = local_quadrature(beta_est_eval);

Afit_beta = [beta_est_eval, beta_est_quad];

coef_beta = Afit_beta \ beta_true_eval;

beta_est_corr = Afit_beta*coef_beta;

rmse_beta = sqrt(mean((beta_true_eval - beta_est_corr).^2));

gain_beta  = sqrt(coef_beta(1)^2 + coef_beta(2)^2);
phase_beta = atan2(coef_beta(2),coef_beta(1));

fprintf('\n=== DEMODULACAO PM COM CORRECAO HILBERT/QUADRATURA ===\n');
fprintf('guard_time   = %.3f us\n', guard_time*1e6);
fprintf('coef_m a     = %.6e\n', coef_m(1));
fprintf('coef_m b     = %.6e\n', coef_m(2));
fprintf('gain_m       = %.6e\n', gain_m);
fprintf('phase_m      = %.6f rad\n', phase_m);
fprintf('RMSE m       = %.6e\n', rmse_m);
fprintf('RMSE m norm  = %.3f %%\n', 100*rmse_norm);
fprintf('coef_beta a  = %.6e\n', coef_beta(1));
fprintf('coef_beta b  = %.6e\n', coef_beta(2));
fprintf('gain_beta    = %.6e\n', gain_beta);
fprintf('phase_beta   = %.6f rad\n', phase_beta);
fprintf('RMSE beta    = %.6e rad\n', rmse_beta);

%% -------------------------------------------------------------------------
% 16. BER / erro de bits, se quiser usar mensagem digital
% -------------------------------------------------------------------------
%
% Para a mensagem senoidal atual, BER nao faz sentido.
% Para onda quadrada, voce pode usar:
%
% bit_true = m_true_eval > 0;
% bit_est  = m_est_corr  > 0;
% BER = mean(bit_true ~= bit_est);
%
% fprintf('BER = %.6e\n', BER);

%% -------------------------------------------------------------------------
% 17. Espectro da fluorescencia integrada pulso a pulso
% -------------------------------------------------------------------------

idx_spec = pulse_t >= t_trans;

Fspec = pulse_F(idx_spec);
Fspec = Fspec - mean(Fspec);

Nfft = length(Fspec);

w = local_hann(Nfft);
Fwin = Fspec.*w;

Y = fft(Fwin);

P2 = abs(Y)/Nfft;
P1 = P2(1:floor(Nfft/2)+1);
P1(2:end-1) = 2*P1(2:end-1);

Fs_p = 1/Tpulse;
f_axis = Fs_p*(0:floor(Nfft/2))/Nfft;

search_bw = max(2e6,0.3*fIF);
idx_band = find(f_axis >= max(0,fIF-search_bw) & f_axis <= fIF+search_bw);

if isempty(idx_band)
    warning('Janela de busca do batimento ficou vazia.');
    f_peak = NaN;
else
    [~,imax_local] = max(P1(idx_band));
    idx_peak = idx_band(imax_local);
    f_peak = f_axis(idx_peak);
end

fprintf('\n=== ESPECTRO PULSO A PULSO ===\n');
fprintf('Pico espectral da fluorescencia integrada = %.3f MHz\n', f_peak/1e6);
fprintf('Batimento esperado                        = %.3f MHz\n', fIF/1e6);
fprintf('Erro relativo                             = %.3f %%\n', 100*abs(f_peak-fIF)/fIF);

%% -------------------------------------------------------------------------
% 18. Plots
% -------------------------------------------------------------------------
%% -------------------------------------------------------------------------
% 18. Plots (IEEE Standard Style)
%% -------------------------------------------------------------------------

% Criar pasta exclusiva para os plots deste código
outDir_pulsado = fullfile(script_dir, 'resultados_pm');
if ~exist(outDir_pulsado, 'dir')
    mkdir(outDir_pulsado);
end

% Função auxiliar em linha para aplicar a padronização e economizar linhas
apply_ieee_style = @(axObj) set(axObj, 'Box', 'on', 'TickDir', 'in', ...
    'TickLength', [0.015 0.015], 'LineWidth', 1.8, 'FontSize', 22, ...
    'FontWeight', 'bold', 'XMinorTick', 'on', 'YMinorTick', 'on');

% 1. Original message
fig1 = figure('Color','w','Position',[80 80 1100 650]); ax = axes('Parent',fig1); hold(ax,'on');
plot(ax, t*1e6, m, '-', 'LineWidth', 4.6); grid(ax,'on'); ax.GridAlpha = 0.12; apply_ieee_style(ax);
xlabel(ax,'Time (\mus)','FontSize',28,'FontWeight','bold'); ylabel(ax,'m(t)','FontSize',28,'FontWeight','bold');
title(ax,'Original message','FontSize',24,'FontWeight','bold');
exportgraphics(fig1, fullfile(outDir_pulsado, '01_original_message.png'), 'Resolution', 600);

% 2. PM phase applied
fig2 = figure('Color','w','Position',[80 80 1100 650]); ax = axes('Parent',fig2); hold(ax,'on');
plot(ax, t*1e6, beta, '-', 'LineWidth', 4.6); grid(ax,'on'); ax.GridAlpha = 0.12; apply_ieee_style(ax);
xlabel(ax,'Time (\mus)','FontSize',28,'FontWeight','bold'); ylabel(ax,'\beta(t) [rad]','FontSize',28,'FontWeight','bold');
title(ax,'PM phase applied to the MW signal','FontSize',24,'FontWeight','bold');
exportgraphics(fig2, fullfile(outDir_pulsado, '02_pm_phase_applied.png'), 'Resolution', 600);

% 3. Laser pulses and detection window
fig3 = figure('Color','w','Position',[80 80 1100 650]); ax = axes('Parent',fig3); hold(ax,'on');
plot(ax, t*1e6, double(laser_on_vec), '-', 'LineWidth', 4.6, 'DisplayName', 'laser ON');
plot(ax, t*1e6, double(detect_on_vec), '.r', 'LineWidth', 4.6, 'DisplayName', 'detection');
grid(ax,'on'); ax.GridAlpha = 0.12; apply_ieee_style(ax); xlim(ax,[0 0.4]);
xlabel(ax,'Time (\mus)','FontSize',28,'FontWeight','bold'); ylabel(ax,'Gate','FontSize',28,'FontWeight','bold');
title(ax,'Laser pulses and detection window','FontSize',24,'FontWeight','bold');
lgd = legend(ax,'Location','best'); set(lgd,'FontSize',20,'FontWeight','bold','Box','on');
exportgraphics(fig3, fullfile(outDir_pulsado, '03_laser_gates.png'), 'Resolution', 600);
%%
% 4. Instantaneous fluorescence
fig4 = figure('Color','w','Position',[80 80 1100 650]); ax = axes('Parent',fig4); hold(ax,'on');
plot(ax, t*1e6, Fluor, '-', 'LineWidth', 4.6);
xline(ax, t_trans*1e6, '--k', 'LineWidth', 2.5, 'DisplayName', 't_{trans}');
xline(ax, Tcal*1e6, '--r', 'LineWidth', 2.5, 'DisplayName', 'PM start');
grid(ax,'on'); ax.GridAlpha = 0.12; apply_ieee_style(ax);
xlabel(ax,'Time (\mus)','FontSize',28,'FontWeight','bold'); ylabel(ax,'Instantaneous fluorescence','FontSize',28,'FontWeight','bold');
title(ax,'Fluorescence','FontSize',24,'FontWeight','bold');
exportgraphics(fig4, fullfile(outDir_pulsado, '04_instantaneous_fluor.png'), 'Resolution', 600);
%%
% 5. Pulse-by-pulse integrated fluorescence
fig5 = figure('Color','w','Position',[80 80 1100 650]); ax = axes('Parent',fig5); hold(ax,'on');
stem(ax, pulse_t*1e6, pulse_F, 'filled', 'LineWidth', 2.5);
xline(ax, t_trans*1e6, '--k', 'LineWidth', 2.5);
xline(ax, Tcal*1e6, '--r', 'LineWidth', 2.5);
grid(ax,'on'); ax.GridAlpha = 0.12; apply_ieee_style(ax);
xlabel(ax,'Time (\mus)','FontSize',28,'FontWeight','bold'); ylabel(ax,'Integrated fluorescence per pulse','FontSize',28,'FontWeight','bold');
title(ax,'Pulse-by-pulse integrated fluorescence','FontSize',24,'FontWeight','bold');
exportgraphics(fig5, fullfile(outDir_pulsado, '05_pulse_integrated_fluor.png'), 'Resolution', 600);

% 6. Spectrum
fig6 = figure('Color','w','Position',[80 80 1100 650]); ax = axes('Parent',fig6); hold(ax,'on');
plot(ax, f_axis/1e6, P1, '-', 'LineWidth', 4.6, 'DisplayName', 'FFT');
xline(ax, fIF/1e6, '--r', 'LineWidth', 2.5, 'DisplayName', 'expected f_{IF}');
if ~isnan(f_peak)
    xline(ax, f_peak/1e6, '--k', 'LineWidth', 2.5, 'DisplayName', 'Detected peak');
end
grid(ax,'on'); ax.GridAlpha = 0.12; apply_ieee_style(ax); xlim(ax,[0, Fs_p/(2e6)]);
xlabel(ax,'Frequency (MHz)','FontSize',28,'FontWeight','bold'); ylabel(ax,'|FFT(F_n)|','FontSize',28,'FontWeight','bold');
title(ax,'Spectrum of pulse-by-pulse integrated fluorescence','FontSize',24,'FontWeight','bold');
lgd = legend(ax,'Location','best'); set(lgd,'FontSize',20,'FontWeight','bold','Box','on');
exportgraphics(fig6, fullfile(outDir_pulsado, '06_fluorescence_spectrum.png'), 'Resolution', 600);

% 7. After DC removal
fig7 = figure('Color','w','Position',[80 80 1100 650]); ax = axes('Parent',fig7); hold(ax,'on');
plot(ax, pulse_t*1e6, F_ac_p, '-', 'LineWidth', 4.6);
xline(ax, Tcal*1e6, '--r', 'LineWidth', 2.5);
grid(ax,'on'); ax.GridAlpha = 0.12; apply_ieee_style(ax);
xlabel(ax,'Time (\mus)','FontSize',28,'FontWeight','bold'); ylabel(ax,'F_{AC,n}','FontSize',28,'FontWeight','bold');
title(ax,'Integrated fluorescence after DC removal','FontSize',24,'FontWeight','bold');
exportgraphics(fig7, fullfile(outDir_pulsado, '07_fluor_after_dc_removal.png'), 'Resolution', 600);

% 8. Lock-in quadratures
fig8 = figure('Color','w','Position',[80 80 1100 650]); ax = axes('Parent',fig8); hold(ax,'on');
plot(ax, pulse_t*1e6, X_lock_p, '-', 'LineWidth', 4.6, 'DisplayName', 'X');
plot(ax, pulse_t*1e6, Y_lock_p, '-', 'LineWidth', 4.6, 'DisplayName', 'Y');
xline(ax, Tcal*1e6, '--r', 'LineWidth', 2.5);
grid(ax,'on'); ax.GridAlpha = 0.12; apply_ieee_style(ax);
xlabel(ax,'Time (\mus)','FontSize',28,'FontWeight','bold'); ylabel(ax,'Quadratures','FontSize',28,'FontWeight','bold');
title(ax,'Pulse-by-pulse lock-in quadratures','FontSize',24,'FontWeight','bold');
lgd = legend(ax,'Location','best'); set(lgd,'FontSize',20,'FontWeight','bold','Box','on');
exportgraphics(fig8, fullfile(outDir_pulsado, '08_lockin_quadratures.png'), 'Resolution', 600);

% 9. Heterodyne phase
fig9 = figure('Color','w','Position',[80 80 1100 650]); ax = axes('Parent',fig9); hold(ax,'on');
plot(ax, pulse_t*1e6, theta_PL_p, '-', 'LineWidth', 4.6);
xline(ax, Tcal*1e6, '--r', 'LineWidth', 2.5);
grid(ax,'on'); ax.GridAlpha = 0.12; apply_ieee_style(ax);
xlabel(ax,'Time (\mus)','FontSize',28,'FontWeight','bold'); ylabel(ax,'\theta_{PL}(t) [rad]','FontSize',28,'FontWeight','bold');
title(ax,'Heterodyne fluorescence phase, pulsed laser','FontSize',24,'FontWeight','bold');
exportgraphics(fig9, fullfile(outDir_pulsado, '09_heterodyne_phase.png'), 'Resolution', 600);

% 10. Beta phase recovery
fig10 = figure('Color','w','Position',[80 80 1100 650]); ax = axes('Parent',fig10); hold(ax,'on');
plot(ax, t_eval*1e6, beta_true_eval, '-', 'LineWidth', 4.6, 'DisplayName', 'original \beta(t)');
plot(ax, t_eval*1e6, beta_est_corr, '--', 'LineWidth', 4.6, 'DisplayName', 'recovered \beta(t)');
grid(ax,'on'); ax.GridAlpha = 0.12; apply_ieee_style(ax);
xlabel(ax,'Time (\mus)','FontSize',28,'FontWeight','bold'); ylabel(ax,'Phase [rad]','FontSize',28,'FontWeight','bold');
title(ax, sprintf('PM phase recovery — RMSE = %.3e rad',rmse_beta),'FontSize',24,'FontWeight','bold');
lgd = legend(ax,'Location','best'); set(lgd,'FontSize',20,'FontWeight','bold','Box','on');
exportgraphics(fig10, fullfile(outDir_pulsado, '10_beta_recovery.png'), 'Resolution', 600);
%%
% 11. Message recovery
fig11 = figure('Color','w','Position',[80 80 1100 650]); ax = axes('Parent',fig11); hold(ax,'on');
plot(ax, t_eval*1e6, m_true_eval, '-', 'LineWidth', 4.6, 'DisplayName', 'original m(t)');
plot(ax, t_eval*1e6, m_est_corr, '--', 'LineWidth', 4.6, 'DisplayName', 'recovered m(t)');
grid(ax,'on'); ax.GridAlpha = 0.12; apply_ieee_style(ax);
xlabel(ax,'Time (\mus)','FontSize',28,'FontWeight','bold'); ylabel(ax,'Message','FontSize',28,'FontWeight','bold');
title(ax, sprintf('PM demodulation — RMSE = %.3e',rmse_m),'FontSize',24,'FontWeight','bold');
lgd = legend(ax,'Location','best'); set(lgd,'FontSize',20,'FontWeight','bold','Box','on');
exportgraphics(fig11, fullfile(outDir_pulsado, '11_message_recovery.png'), 'Resolution', 600);

%% --- Digital Plots ---

% 12. Applied digital message
fig12 = figure('Color','w','Position',[80 80 1100 650]); ax = axes('Parent',fig12); hold(ax,'on');
stairs(ax, t*1e6, m, 'LineWidth', 4.6, 'DisplayName', 'applied m(t)');
stem(ax, bit_center_t*1e6, symbols_tx, 'filled', 'LineWidth', 2.5, 'DisplayName', 'bit center');
grid(ax,'on'); ax.GridAlpha = 0.12; apply_ieee_style(ax);
xlabel(ax,'Time (\mus)','FontSize',28,'FontWeight','bold'); ylabel(ax,'m(t) / symbols','FontSize',28,'FontWeight','bold');
title(ax,'Applied digital message','FontSize',24,'FontWeight','bold');
lgd = legend(ax,'Location','best'); set(lgd,'FontSize',20,'FontWeight','bold','Box','on');
exportgraphics(fig12, fullfile(outDir_pulsado, '12_applied_digital_message.png'), 'Resolution', 600);
%%
% 13. Bit decision
fig13 = figure('Color','w','Position',[80 80 1100 650]); ax = axes('Parent',fig13); hold(ax,'on');
stairs(ax, bit_center_t*1e6, symbols_tx, 'LineWidth', 4.6, 'DisplayName', 'transmitted symbol');
plot(ax, bit_center_t(valid_bits)*1e6, score_bits(valid_bits), 'o--', 'LineWidth', 3.0, 'DisplayName', 'demodulated score');
xline(ax, 0, '--k', 'LineWidth', 2.5, 'DisplayName', 'threshold');
grid(ax,'on'); ax.GridAlpha = 0.12; apply_ieee_style(ax);
xlabel(ax,'Time (\mus)','FontSize',28,'FontWeight','bold'); ylabel(ax,'Decision score','FontSize',28,'FontWeight','bold');
title(ax, sprintf('Bit decision — Test BER = %.3g',BER_test),'FontSize',24,'FontWeight','bold');
lgd = legend(ax,'Location','best'); set(lgd,'FontSize',20,'FontWeight','bold','Box','on');
exportgraphics(fig13, fullfile(outDir_pulsado, '13_bit_decision_score.png'), 'Resolution', 600);
%%
% 14. Bit-by-bit comparison
fig14 = figure('Color','w','Position',[80 80 1100 650]); ax = axes('Parent',fig14); hold(ax,'on');
stem(ax, 1:Nbits, bits_tx, 'filled', 'LineWidth', 3.0, 'DisplayName', 'TX bits');
stem(ax, find(valid_bits), bits_rx(valid_bits), 'LineWidth', 2.0, 'DisplayName', 'RX bits');
grid(ax,'on'); ax.GridAlpha = 0.12; apply_ieee_style(ax);
xlabel(ax,'Bit index','FontSize',28,'FontWeight','bold'); ylabel(ax,'Bit','FontSize',28,'FontWeight','bold');
title(ax, sprintf('Bit-by-bit comparison — Total BER = %.3e',BER_all),'FontSize',24,'FontWeight','bold');
lgd = legend(ax,'Location','best'); set(lgd,'FontSize',20,'FontWeight','bold','Box','on');
exportgraphics(fig14, fullfile(outDir_pulsado, '14_bit_comparison.png'), 'Resolution', 600);

% 15. Bit errors
fig15 = figure('Color','w','Position',[80 80 1100 650]); ax = axes('Parent',fig15); hold(ax,'on');
stem(ax, find(valid_bits), bit_errors_all, 'filled', 'LineWidth', 3.0);
grid(ax,'on'); ax.GridAlpha = 0.12; apply_ieee_style(ax); ylim(ax, [-0.1 1.1]);
xlabel(ax,'Bit index','FontSize',28,'FontWeight','bold'); ylabel(ax,'Error','FontSize',28,'FontWeight','bold');
title(ax, sprintf('Bit errors — %d errors in %d valid bits',Nerr_all,Nvalid_all),'FontSize',24,'FontWeight','bold');
exportgraphics(fig15, fullfile(outDir_pulsado, '15_bit_errors.png'), 'Resolution', 600);
%% -------------------------
% Salvar dados para o Python
%% -------------------------
out_PM.t            = t;
out_PM.Fluor        = Fluor;
out_PM.pulse_t      = pulse_t;
out_PM.pulse_F      = pulse_F;
out_PM.Tpulse       = Tpulse;
out_PM.t_laser      = t_laser;
out_PM.t_detect     = t_detect;
out_PM.Tcal         = Tcal;
out_PM.Tbit         = Tbit;
out_PM.Nbits        = Nbits;
out_PM.bits_tx      = bits_tx;
out_PM.bits_rx      = bits_rx;
out_PM.bit_center_t = bit_center_t;
out_PM.theta_PL_p   = theta_PL_p;
out_PM.beta_rec_p   = beta_rec_p;
out_PM.dt           = dt;
out_PM.rho01 = rho01;
out_PM.phi01 = phi01;
out_PM.BER_test = BER_test;
out_PM.BER_train = BER_train;
out_PM.valid_bits = valid_bits;
out_PM.nv_rates = nv_rates;
out_PM.R_532_on = R_532;
save(fullfile(outDir_pulsado, 'out_PM.mat'), 'out_PM');

function w = local_hann(n)
    if n == 1
        w = 1;
    else
        w = 0.5 - 0.5*cos(2*pi*(0:n-1)'/(n-1));
    end
end

function y = local_zero_phase_fir(b, x)
    % Duas passagens FIR com reflexao impar nas bordas e ganho DC unitario.
    p = 3*(numel(b)-1);
    assert(numel(x) > p, 'A serie deve ter mais de %d amostras para este filtro.', p);
    left = 2*x(1) - x(p+1:-1:2);
    right = 2*x(end) - x(end-1:-1:end-p);
    xpad = [left; x; right];
    y = filter(b, 1, xpad - xpad(1)) + xpad(1);
    y = flipud(y);
    y = filter(b, 1, y - y(1)) + y(1);
    y = flipud(y);
    y = y(p+1:p+numel(x));
end

function q = local_quadrature(x)
    n = numel(x);
    h = zeros(n,1);
    h(1) = 1;
    h(2:floor((n+1)/2)) = 2;
    if mod(n,2) == 0
        h(n/2+1) = 1;
    end
    q = imag(ifft(fft(x(:)).*h));
end