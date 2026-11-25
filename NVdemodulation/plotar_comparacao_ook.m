function recovery = plotar_comparacao_ook(use_total_flux)
%PLOTAR_COMPARACAO_OOK Sobrepõe as simulações de AM/OOK com e sem cavidade.
% Execute antes simular_ook_sem_cavidade.m e simular_ook_com_cavidade.m.
% Opcional: rode simular_ook_com_cavidade.m com zpl_fwhm = 0 para gerar a
% referência sem defasamento da ZPL; se existir, ela entra como 3a curva.
%
% Por padrão (true) compara o fluxo óptico TOTAL nos dois casos, que é a
% grandeza fisicamente comparável. Passe false para usar a saída superior
% de M2 com cavidade (diagnóstico; só faz sentido na figura normalizada).
%
% Figuras geradas:
%   comparison_absolute      fluxo absoluto (mostra o quanto a cavidade muda o nível)
%   comparison_IEEE          visão normalizada min-max
%   rise_time_explained      recuperação 10-90 % após desligar a MW

arguments
    use_total_flux (1,1) logical = true
end

base_dir  = fileparts(mfilename('fullpath'));
cav_file  = fullfile(base_dir,'resultados_ook_com_cavidade','out_cav.mat');
free_file = fullfile(base_dir,'resultados_ook_sem_cavidade','out_noCav.mat');
ref_file  = fullfile(base_dir,'resultados_ook_com_cavidade_sem_defasamento','out_cav.mat');
assert(isfile(cav_file) && isfile(free_file), ...
    'Execute simular_ook_com_cavidade.m e simular_ook_sem_cavidade.m nesta pasta.');

cav  = getfield(load(cav_file,'out_cav'),'out_cav');
free = getfield(load(free_file,'out_noCav'),'out_noCav');
assert(isequal(cav.mw_bits(:),free.mw_bits(:)) && cav.Tbit == free.Tbit, ...
    'As simulações precisam usar a mesma sequência e duração dos bits.');

% Séries: sem cavidade, com cavidade e (opcional) referência sem defasamento
series = struct('t',{},'F',{},'on',{},'off',{},'label',{},'color',{},'style',{});
series(end+1) = struct('t',free.t,'F',free.F,'on',free.F_on_ss,'off',free.F_off_ss, ...
    'label',"Without cavity (free space)",'color',[0.85 0.33 0.10],'style','-');

if use_total_flux
    cav_trace = cav.F;  cav_on = cav.F_on_ss;  cav_off = cav.F_off_ss;
    cav_label = "With cavity (total flux)";
    out_dir = fullfile(base_dir,'resultados_comparacao_ook_fluxo_total');
else
    assert(isfield(cav,'I_M2_top_on_ss'), ...
        'Gere novamente out_cav.mat com a versão atual de simular_ook_com_cavidade.m.');
    cav_trace = cav.I_M2_top;  cav_on = cav.I_M2_top_on_ss;  cav_off = cav.I_M2_top_off_ss;
    cav_label = "With cavity (M2 top)";
    out_dir = fullfile(base_dir,'resultados_comparacao_ook');
end
if isfield(cav,'zpl_fwhm') && cav.zpl_fwhm > 0
    cav_label = cav_label + sprintf(", ZPL FWHM %.2g THz", cav.zpl_fwhm/1e12);
end
series(end+1) = struct('t',cav.t,'F',cav_trace,'on',cav_on,'off',cav_off, ...
    'label',cav_label,'color',[0.00 0.45 0.74],'style','--');

if isfile(ref_file)
    ref = getfield(load(ref_file,'out_cav'),'out_cav');
    if isequal(ref.mw_bits(:),cav.mw_bits(:)) && ref.Tbit == cav.Tbit
        if use_total_flux
            rt = ref.F;  ron = ref.F_on_ss;  roff = ref.F_off_ss;
        else
            rt = ref.I_M2_top;  ron = ref.I_M2_top_on_ss;  roff = ref.I_M2_top_off_ss;
        end
        series(end+1) = struct('t',ref.t,'F',rt,'on',ron,'off',roff, ...
            'label',"With cavity, no ZPL dephasing (reference)",'color',[0.45 0.45 0.45],'style',':');
    else
        warning('Referência sem defasamento ignorada: sequência de bits diferente.');
    end
end
ns = numel(series);

% Malha comum
t0 = max(arrayfun(@(s) s.t(1), series));
t1 = min(arrayfun(@(s) s.t(end), series));
t  = cav.t(cav.t >= t0 & cav.t <= t1);
assert(numel(t) > 1, 'As séries precisam ter um intervalo de tempo comum.');
flux = zeros(numel(t), ns);
for k = 1:ns
    flux(:,k) = interp1(series(k).t, series(k).F, t, 'linear');
end
assert(all(isfinite(flux),'all'), 'Os fluxos devem ter valores finitos.');

if ~isfolder(out_dir)
    mkdir(out_dir);
end

% ---------- (1) Fluxo absoluto ----------
[fig,ax] = comparison_axes([80 80 1100 650]);
for k = 1:ns
    plot(ax, t*1e6, flux(:,k)/1e6, series(k).style, 'Color', series(k).color, ...
        'LineWidth', 4.6, 'DisplayName', series(k).label);
end
bit_edges = find(diff(double(cav.mw_bits(:))) ~= 0)*cav.Tbit;
for x = bit_edges.'
    xline(ax, x*1e6, ':k', 'LineWidth', 1.5, 'HandleVisibility', 'off');
end
xlabel(ax,'Time (\mus)','FontSize',28,'FontWeight','bold');
ylabel(ax,'Optical flux (10^6 s^{-1})','FontSize',28,'FontWeight','bold');
title(ax,'AM/OOK: absolute optical flux','FontSize',24,'FontWeight','bold');
xlim(ax,[t(1),t(end)]*1e6);
style_legend(ax,'southwest');
save_plot(fig,out_dir,'comparison_absolute');

% ---------- (2) Visão normalizada min-max ----------
low  = min(flux,[],1);
span = max(flux,[],1) - low;
normalized = (flux - low)./max(span, realmin);
[fig,ax] = comparison_axes([80 80 1100 650]);
for k = 1:ns
    plot(ax, t*1e6, normalized(:,k), series(k).style, 'Color', series(k).color, ...
        'LineWidth', 4.6, 'DisplayName', series(k).label);
end
xlabel(ax,'Time (\mus)','FontSize',28,'FontWeight','bold');
ylabel(ax,'Normalized intensity','FontSize',28,'FontWeight','bold');
title(ax,'Normalized AM/OOK comparison','FontSize',24,'FontWeight','bold');
xlim(ax,[t(1),t(end)]*1e6);
ylim(ax,[-0.02,1.02]);
style_legend(ax,'north');
save_plot(fig,out_dir,'comparison_IEEE');

% ---------- (3) Recuperação 10-90 % após desligar a MW ----------
bits    = double(cav.mw_bits(:));
off_bit = find(diff(bits) < 0, 1);
recovery = table();
if isempty(off_bit)
    warning('A sequência precisa de uma borda 1 -> 0 para medir a recuperação.');
    return
end
start   = max(off_bit*cav.Tbit, t(1));
next_on = find(diff(bits) > 0 & (1:numel(bits)-1)' > off_bit, 1);
stop    = t(end);
if ~isempty(next_on)
    stop = min(stop, next_on*cav.Tbit);
end
edge_time = [start; t(t > start & t < stop); stop];
edge_ns   = (edge_time - off_bit*cav.Tbit)*1e9;
edge_flux = interp1(t, flux, edge_time, 'linear');

on_lv  = [series.on];  off_lv = [series.off];
response = (edge_flux - on_lv)./(off_lv - on_lv);

t10 = nan(ns,1);  t90 = nan(ns,1);
[fig,ax] = comparison_axes([80 80 1100 850]);
for k = 1:ns
    t10(k) = first_crossing(edge_ns, response(:,k), 0.10);
    t90(k) = first_crossing(edge_ns, response(:,k), 0.90);
end
yline(ax,0.10,':','10%','LineWidth',1.5,'HandleVisibility','off');
yline(ax,0.90,':','90%','LineWidth',1.5,'HandleVisibility','off');
for k = 1:ns
    name = series(k).label;
    if isfinite(t10(k)) && isfinite(t90(k))
        name = sprintf('%s (%.0f ns)', name, t90(k) - t10(k));
    end
    plot(ax, edge_ns, response(:,k), series(k).style, 'Color', series(k).color, ...
        'LineWidth', 4.6, 'DisplayName', name);
end
xlabel(ax,'Time after microwave turn-off (ns)','FontSize',28,'FontWeight','bold');
ylabel(ax,'Normalized fluorescence recovery','FontSize',28,'FontWeight','bold');
title(ax,'10–90% recovery time comparison','FontSize',24,'FontWeight','bold');
xlim(ax,[edge_ns(1), edge_ns(end)]);
ylim(ax,[-0.02, 1.06]);
style_legend(ax,'southeast');
save_plot(fig,out_dir,'rise_time_explained');

contrast = (off_lv - on_lv)./off_lv;
recovery = table([series.label].', off_lv.'/1e6, on_lv.'/1e6, 100*contrast.', t10, t90, t90 - t10, ...
    'VariableNames',{'Signal','F_off_1e6','F_on_1e6','Contrast_pct','t10_ns','t90_ns','Recovery_10_90_ns'});
disp(recovery);
writetable(recovery, fullfile(out_dir,'resumo_comparacao.csv'));
fprintf('Figuras salvas em %s\n', out_dir);
end

function [fig,ax] = comparison_axes(position)
fig = figure('Color','w','Position',position);
ax = axes(fig);
hold(ax,'on');  box(ax,'on');  grid(ax,'on');
set(ax,'TickDir','in','TickLength',[0.015 0.015],'LineWidth',1.8, ...
    'FontSize',22,'FontWeight','bold','XMinorTick','on','YMinorTick','on', ...
    'GridAlpha',0.12);
end

function style_legend(ax,location)
lgd = legend(ax,'Location',location);
set(lgd,'FontSize',18,'FontWeight','bold','Box','on');
end

function save_plot(fig,out_dir,name)
exportgraphics(fig,fullfile(out_dir,[name '.pdf']),'ContentType','vector');
exportgraphics(fig,fullfile(out_dir,[name '.png']),'Resolution',600);
end

function t_event = first_crossing(t,y,threshold)
idx = find(y >= threshold, 1);
if isempty(idx)
    t_event = NaN;
elseif idx == 1
    t_event = t(1);
else
    fraction = (threshold - y(idx-1))/(y(idx) - y(idx-1));
    t_event = t(idx-1) + fraction*(t(idx) - t(idx-1));
end
end