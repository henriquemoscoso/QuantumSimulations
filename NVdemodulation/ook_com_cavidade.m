%% AM/OOK com dois modos de cavidade e dinâmica explícita de Jaynes--Cummings
% A comparação com simular_ook_sem_cavidade.m usa a mesma sequência, as
% mesmas taxas do NV e o mesmo bombeio incoerente em 532 nm.
% Depois de gerar os dois arquivos MAT, execute plotar_comparacao_ook.
clear; clc; close all;
script_dir = fileparts(mfilename('fullpath'));

enable_cavity = true;
Bz = 5e-3;                       % T
gamma_e = 2*pi*28e9;              % rad/(s T)
Dg = 2*pi*2.87e9;                 % rad/s
f_mw = 3.000e9;                   % Hz
S_mw_on = 1e8;                    % W/m^2, entrada do modelo
B_mw = sqrt(4*pi*1e-7*S_mw_on/299792458);
Omega_R = gamma_e*B_mw/sqrt(2);   % rad/s, escala ressonante por transição

P_532 = 40e-3;                    % W
k_532 = 1e10;                      % s^-1/W, coeficiente do modelo
R_532 = k_532*P_532;              % s^-1, bomba CW e incoerente
Tbit = 2e-6;                      % >= 3x a recuperação 10-90 (~0.6 us)
dt = 0.1e-9;
mw_bits = logical([0; 1; 1; 0]);
steps_per_bit = round(Tbit/dt);
assert(abs(steps_per_bit*dt - Tbit) < 1e-8*dt, ...
    'Tbit deve ser um múltiplo inteiro de dt.');

% Taxas em s^-1. Tayefeh Younesi et al., Tabela I:
% https://doi.org/10.1103/PhysRevApplied.23.054019
Gamma_r = 66e6;
Gamma_isc = [53e6, 7.9e6, 53e6];

% Taxas por subnível. Médias da Tabela 1 de Tetienne et al.:
% https://doi.org/10.1088/1367-2630/14/10/103033
Gamma_sg = [0.73e6, 0.98e6, 0.73e6];
tau_ue = 100e-15;                % escala efetiva, Ulbricht et al. (2018)
Gamma_ue = 1/tau_ue;             % https://doi.org/10.1103/PhysRevB.97.220302

% Parâmetros dos dois modos de polarização do estudo de Yurgens et al.:
% https://doi.org/10.1038/s41534-024-00915-9
% g do artigo é estimado para M2. M1 usa a projeção relativa pelo ângulo.
xi0 = 0.03;
kappa_number = 2*pi*11e9;        % taxa de perda de número, s^-1
delta_M1 = 2*pi*34.4e9;          % frequência de M1 em relação a M2, rad/s
theta_cavity = deg2rad(33);
g2 = 2*pi*167e6;
g1 = g2*tan(theta_cavity);
top_fraction = 0.90;             % escolha de coleta, para diagnóstico
% g e kappa vêm de um experimento criogênico. À temperatura ambiente a
% coerência óptica e-g sofre defasamento puro pelos fônons, alargando a ZPL
% para a ordem de THz. Sem esse termo, a taxa via cavidade (~4g^2/kappa)
% fica superestimada por uma ou mais ordens de grandeza.
% VERIFICAR: largura da ZPL a 300 K na literatura e ajustar zpl_fwhm.
zpl_fwhm = 1e12;                 % Hz, FWHM da ZPL a 300 K (ordem de grandeza)
                                 % use 0 para gerar a referência sem defasamento
gamma_star = pi*zpl_fwhm;        % s^-1, decaimento da coerência e-g

% Base NV: g(+1,0,-1), e(+1,0,-1), u(+1,0,-1), s.
g = 1:3;
e = 4:6;
u = 7:9;
s = 10;
dNV = 10;
g0 = g(2);
e0 = e(2);

% RWA. O nível g,0 é a referência do setor fundamental.
omega_mw = 2*pi*f_mw;
H_NV = sparse(diag([Dg+gamma_e*Bz-omega_mw, 0, ...
                    Dg-gamma_e*Bz-omega_mw, zeros(1,7)]));
H_MW_NV = (Omega_R/2)*sparse([1,2,2,3],[2,1,3,2],1,dNV,dNV);

nF = 1 + double(enable_cavity);
d = nF*nF*dNV;
IN = speye(dNV);
IF = speye(nF);
IF2 = speye(nF*nF);
if enable_cavity
    a = sparse(1,2,1,2,2);       % truncamento em no máximo um fóton por modo
    a1 = kron(kron(a,IF),IN);
    a2 = kron(kron(IF,a),IN);
    % Os três ramos e_m -> g_m têm a mesma frequência óptica e o mesmo
    % dipolo, então todos se acoplam aos modos da cavidade.
    lower_NV = kron(IF2,sparse(g,e,1,dNV,dNV));
    coupling = g1*(a1'*lower_NV) + g2*(a2'*lower_NV);
    H_cavity = delta_M1*(a1'*a1) + coupling + coupling';
else
    a1 = sparse(d,d);
    a2 = sparse(d,d);
    H_cavity = sparse(d,d);
end

H_off = kron(IF2,H_NV) + H_cavity;
H_on = H_off + kron(IF2,H_MW_NV);

% Cada ciclo radiativo e -> g conserva o spin. As frações ZPL e PSB
% compartilham o mesmo operador de salto no espaço NV.
jumps = {};
for m = 1:3
    jumps{end+1} = kron(IF2,sparse(u(m),g(m),sqrt(R_532),dNV,dNV));
    jumps{end+1} = kron(IF2,sparse(e(m),u(m),sqrt(Gamma_ue),dNV,dNV));
    jumps{end+1} = kron(IF2,sparse(g(m),e(m),sqrt(Gamma_r),dNV,dNV));
    jumps{end+1} = kron(IF2,sparse(s,e(m),sqrt(Gamma_isc(m)),dNV,dNV));
    jumps{end+1} = kron(IF2,sparse(g(m),s,sqrt(Gamma_sg(m)),dNV,dNV));
end
% Defasamento puro da coerência óptica e-g (só atua quando há cavidade,
% pois sem ela nenhuma coerência e-g é criada; mantido nos dois modelos).
sigma_z_opt = sparse([e,g],[e,g],[ones(1,3),-ones(1,3)],dNV,dNV);
jumps{end+1} = kron(IF2,sqrt(gamma_star/2)*sigma_z_opt);
if enable_cavity
    jumps{end+1} = sqrt(kappa_number)*a1;
    jumps{end+1} = sqrt(kappa_number)*a2;
end

L_diss = sparse(d*d,d*d);
for idx = 1:numel(jumps)
    op = jumps{idx};
    loss = op'*op;
    L_diss = L_diss + kron(conj(op),op) ...
        - 0.5*kron(speye(d),loss) - 0.5*kron(loss.',speye(d));
end
L_off = L_diss + commutator(H_off);
L_on = L_diss + commutator(H_on);

% Os mapas só precisam das entradas de rho alcançáveis a partir das
% populações. A redução estrutural é exata para os operadores acima.
diagonal = (1:d+1:d*d)';
active = false(d*d,1);
active(diagonal) = true;
coupled = spones(L_on) | spones(L_off);
while true
    expanded = active | any(coupled(:,active),2);
    if isequal(active,expanded)
        break
    end
    active = expanded;
end
L_off = full(L_off(active,active));
L_on = full(L_on(active,active));

diag_inside_active = find(ismember(find(active),diagonal));
nv_state = repmat((1:dNV)', nF*nF, 1);
weights = [
    ones(1,d)
    (nv_state==g0).'
    (nv_state==s).'
    Gamma_r*ismember(nv_state,e).'
    kappa_number*full(diag(a1'*a1)).'
    kappa_number*full(diag(a2'*a2)).'
];
readout = zeros(size(weights,1),size(L_off,1));
readout(:,diag_inside_active) = weights;

% Começa no regime estacionário com MW desligada para medir a resposta
% a degrau com a mesma condição inicial nos dois modelos.
rho_off = stationary_state(L_off,readout(1,:));
rho_on = stationary_state(L_on,readout(1,:));
flux_selector = sum(readout(4:6,:),1);
F_off = real(flux_selector*rho_off);
F_on = real(flux_selector*rho_on);
I_M2_top_off_ss = top_fraction*real(readout(6,:)*rho_off);
I_M2_top_on_ss = top_fraction*real(readout(6,:)*rho_on);
U_off = expm(L_off*dt);
U_on = expm(L_on*dt);

Nsteps = numel(mw_bits)*steps_per_bit;
mw_samples = repelem(mw_bits,steps_per_bit);
t = (0:Nsteps)'*dt;
trace_values = zeros(size(weights,1),Nsteps+1);
rho = rho_off;
for k = 1:Nsteps
    trace_values(:,k) = real(readout*rho);
    if mw_samples(k)
        rho = U_on*rho;
    else
        rho = U_off*rho;
    end
end
trace_values(:,end) = real(readout*rho);
trace_error = max(abs(trace_values(1,:)-1));
assert(trace_error < 1e-6 && all(isfinite(trace_values),'all'), ...
    'Propagação instável. Erro máximo de traço: %.3g.', trace_error);

% Fluxo óptico total em ambos os casos. M2 pela saída superior é um
% diagnóstico adicional e fica disponível em I_M2_top.
F_free = trace_values(4,:).';
F_M1 = trace_values(5,:).';
F_M2 = trace_values(6,:).';
F = F_free + F_M1 + F_M2;
I_M2_top = top_fraction*F_M2;

% Tempos de 10% a 90% do degrau, relativos aos níveis estacionários.
% Usa o primeiro cruzamento de cada limiar, mesmo com sobre-elevação.
t_on = first_step_duration(t,F,mw_bits,steps_per_bit,true,F_off,F_on);
t_off = first_step_duration(t,F,mw_bits,steps_per_bit,false,F_on,F_off);

% rise_time: subida da fluorescência quando a MW desliga (bit 1 -> 0).
%   rise_time = t(90 %) - t(10 %), com os níveis definidos pelo degrau entre
%   o estacionário com MW ligada (F_on) e o estacionário com MW desligada (F_off).
% fall_time: queda correspondente quando a MW liga (bit 0 -> 1).
rise_time = t_off;
fall_time = t_on;
if isnan(rise_time)
    warning(['A fluorescência não atingiu 90 %% do estacionário dentro do bit. ' ...
             'Aumente Tbit para medir o rise_time.']);
end
contrast = (F_off-F_on)/F_off;

result = struct('t',t,'F',F,'F_free',F_free,'F_ZPL_free',xi0*F_free, ...
    'F_PSB',(1-xi0)*F_free,'F_M1',F_M1,'F_M2',F_M2,'I_M2_top',I_M2_top, ...
    'I_M2_top_on_ss',I_M2_top_on_ss,'I_M2_top_off_ss',I_M2_top_off_ss, ...
    'Pg0',trace_values(2,:).','Ps',trace_values(3,:).','trace_error',trace_error, ...
    'dt',dt,'Tbit',Tbit,'mw_bits',mw_bits,'F_on_ss',F_on,'F_off_ss',F_off, ...
    't_10_90_mw_on',t_on,'t_10_90_mw_off',t_off,'contrast',contrast, ...
    'rise_time',rise_time,'fall_time',fall_time, ...
    'nv_rates',struct('Gamma_r',Gamma_r,'Gamma_isc0',Gamma_isc(2), ...
        'Gamma_isc_pm',Gamma_isc(1),'Gamma_sg0',Gamma_sg(2), ...
        'Gamma_sgpm_each',Gamma_sg(1),'tau_s',1/sum(Gamma_sg), ...
        'p_s0',Gamma_sg(2)/sum(Gamma_sg),'tau_ue',tau_ue, ...
        'tau_e0',1/(Gamma_r+Gamma_isc(2)),'tau_epm',1/(Gamma_r+Gamma_isc(1))), ...
    'Bz',Bz,'f_mw',f_mw,'Omega_R',Omega_R,'R_532',R_532, ...
    'cavity_enabled',enable_cavity,'kappa_number',enable_cavity*kappa_number, ...
    'nF',nF,'g1',enable_cavity*g1,'g2',enable_cavity*g2,'xi0',xi0, ...
    'gamma_star',gamma_star,'zpl_fwhm',zpl_fwhm);

if enable_cavity
    label = 'With cavity';
    out_dir = fullfile(script_dir,'resultados_ook_com_cavidade');
    if zpl_fwhm == 0              % referência sem defasamento da ZPL
        out_dir = fullfile(script_dir,'resultados_ook_com_cavidade_sem_defasamento');
        label = 'With cavity (no ZPL dephasing)';
    end
    out_cav = result;
    out_cav.I = I_M2_top;        % campo legado para gráficos de M2
    result_file = 'out_cav.mat';
    result_variable = 'out_cav';
else
    label = 'Without cavity';
    out_dir = fullfile(script_dir,'resultados_ook_sem_cavidade');
    out_noCav = result;
    result_file = 'out_noCav.mat';
    result_variable = 'out_noCav';
end
if ~isfolder(out_dir)
    mkdir(out_dir);
end
save(fullfile(out_dir,result_file),result_variable);

fprintf('\n%s\n',label);
fprintf('Rabi / 2pi = %.3f MHz, R_532 = %.2e /s\n',Omega_R/(2*pi*1e6),R_532);
fprintf('Contraste estacionário = %.3f %%\n',100*contrast);
if enable_cavity
    fprintf('Taxa via M2 estimada 4g^2/(kappa+2gamma*) = %.3g /s (sem defasamento: %.3g /s)\n', ...
        4*g2^2/(kappa_number + 2*gamma_star), 4*g2^2/kappa_number);
end
fprintf('Rise time (MW desliga, PL sobe, 10-90%%) = %.2f ns\n',rise_time*1e9);
fprintf('Fall time (MW liga, PL desce, 10-90%%)   = %.2f ns\n',fall_time*1e9);
fprintf('Resultados em %s\n',out_dir);

fig = figure('Color','w','Name',label);
layout = tiledlayout(fig,2,1);
ax = nexttile(layout);
stairs(ax,(0:numel(mw_bits))*Tbit*1e6,double([mw_bits;mw_bits(end)]),'LineWidth',1.5);
ylabel(ax,'Microwave state');
ylim(ax,[-0.1,1.1]);
grid(ax,'on');
ax = nexttile(layout);
plot(ax,t*1e6,F/1e6,'LineWidth',1.5,'DisplayName','Total optical flux');
hold(ax,'on');
if enable_cavity
    plot(ax,t*1e6,I_M2_top/1e6,'LineWidth',1.2,'DisplayName','M2 top flux');
end
yline(ax,F_off/1e6,':','HandleVisibility','off');
yline(ax,F_on/1e6,':','HandleVisibility','off');
xlabel(ax,'Time (\mus)');
ylabel(ax,'Photon flux (10^6 s^{-1})');
legend(ax,'Location','best');
grid(ax,'on');
title(layout,['AM/OOK — ' label]);
exportgraphics(fig,fullfile(out_dir,'ook_fluxo_optico.png'),'Resolution',300);

function L = commutator(H)
    I = speye(size(H,1));
    L = -1i*(kron(I,H) - kron(H.',I));
end

function rho = stationary_state(L, trace_row)
    system = L/max(abs(L),[],'all');
    row = find(trace_row,1);
    system(row,:) = trace_row;
    rhs = zeros(size(L,1),1);
    rhs(row) = 1;
    rho = system\rhs;
end

function duration = first_step_duration(t, F, bits, samples_per_bit, on, initial, final)
    edges = find(diff([false;bits]) ~= 0);
    edge = find(bits(edges)==on,1);
    if isempty(edge) || abs(final-initial) <= 1e-9*max(abs([initial,final]))
        duration = NaN;
        return
    end
    stops = [edges(2:end);numel(bits)+1];
    samples = (edges(edge)-1)*samples_per_bit+1:(stops(edge)-1)*samples_per_bit+1;
    progress = (F(samples)-initial)/(final-initial);
    duration = first_crossing(t(samples),progress,0.90) ...
             - first_crossing(t(samples),progress,0.10);
end

function time = first_crossing(t, progress, level)
    hit = find(progress >= level,1);
    if isempty(hit)
        time = NaN;
    elseif hit == 1
        time = t(1);
    else
        fraction = (level-progress(hit-1))/(progress(hit)-progress(hit-1));
        time = t(hit-1) + fraction*(t(hit)-t(hit-1));
    end
end