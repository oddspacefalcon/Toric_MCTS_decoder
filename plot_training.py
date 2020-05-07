import matplotlib.pyplot as plt
import numpy as np


d = 11
plot_name = f'd={d}'
file_path = f'data/d_{d}/data_all.txt'

win_rate_list = []
margin = 10
p_errors = [0] * margin

with open(file_path) as f:
    l1 = f.readline().split(',')
    w_index = l1.index(' win_rate')
    p_index = l1.index(' p_error_train')
    t_index = l1.index(' training_time\n')

    f.readline()
    for line in f:
        l = line.split(',')
        win_rate_list.append(float(l[w_index]))
        p_errors.append(round(float(l[p_index]), 3))

    time = float(l[t_index]) / 3600

epoker = list(range(len(win_rate_list)))

inds = []
p = set()
for p_e in p_errors:
    if p_e not in p and p_e != 0:
        inds.append(p_errors.index(p_e))
        p.add(p_e)
p_end = p_e
p_errors.extend([0] * margin)
inds.append(len(p_errors))

i = 0
window_size = 20
moving_average = []
while i < len(win_rate_list) - window_size + 1:

    moving_average.append(sum(win_rate_list[i: i + window_size]) / window_size)
    i += 1

t_plot = list(np.arange(0, time, time / len(win_rate_list)))
step = 40
t_plot = [round(t, 1) for t in t_plot][::step]

fig, ax = plt.subplots(figsize=(12, 8))

ax.scatter(epoker, win_rate_list, label='1000 felkorringeringar')
ax.scatter(epoker[:len(win_rate_list) - window_size + 1], moving_average, label='Glidande medelvärde, 20000 felkorringeringar')
ax.legend(fontsize=14)

ax2 = ax.twiny()
ticks = inds[len(inds) - 2:]
ticks.insert(0, 0)
ax2.set_xticks(ticks)
ax2.set_xticklabels([f'$P_e$ = 0.01', f'$P_e$ = {p_end}'])
#ax2.set_xlabel('$P_e$', fontsize=15)

ax.set_xticks(epoker[::step])
ax.set_xticklabels(t_plot)
ax.set_xlim(-margin, len(win_rate_list)  + margin)

ax.tick_params(axis='both', labelsize=14)
ax2.tick_params(axis='both', labelsize=14)

ax.set_title(f'Träning, d = {d}', fontsize=20)
ax.set_ylabel('Andel lyckade korrigeringar', fontsize=20)
ax.set_xlabel('Tid, [h]', fontsize=20)

plt.savefig('plots/' + plot_name + '.png')
