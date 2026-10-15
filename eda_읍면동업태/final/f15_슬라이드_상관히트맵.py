# 발표용 변수 간 상관 히트맵 (음식점 칸, 학습 2023~2025, Spearman)
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
mpl.rcParams['font.family'] = 'WenQuanYi Zen Hei'
mpl.rcParams['axes.unicode_minus'] = False
d = pd.read_csv('/mnt/project-files/eda/인천_분기x읍면동x업태_다음분기폐업률.csv')
t = d[(d.기준연도 <= 2025) & (d.업태구분명 == '음식점')]
cols = {'읍면동업태_영업점포수': '점포수', '읍면동업태_직전분기말점포수': '직전분기말 점포수',
        '읍면동업태_개업률': '개업률', '읍면동업태_폐업률': '폐업률', '읍면동업태_점포수증감률': '점포수 증감률',
        '읍면동업태_폐업률_4Q평균': '폐업률 4Q평균', '읍면동_폐업률': '읍면동_폐업률',
        '평균업력개월': '평균업력', '다음분기_폐업률': '다음분기 폐업률'}
c = t[list(cols)].corr('spearman').values
labels = list(cols.values()); n = len(labels)
cmap = LinearSegmentedColormap.from_list('div', ['#eb6834', '#f3f2ee', '#0067ff'])
fig, ax = plt.subplots(figsize=(9, 8.4), dpi=170); fig.patch.set_facecolor('white')
sub = c[1:, :-1]
m = np.ma.array(sub, mask=np.triu(np.ones_like(sub, bool), 1))
ax.imshow(m, cmap=cmap, vmin=-1, vmax=1)
hi = {(1, 0), (6, 3), (4, 3)}
for i in range(1, n):
    for j in range(i):
        v = c[i, j]
        ax.text(j, i - 1, f'{v:.2f}', ha='center', va='center', fontsize=14,
                color='white' if abs(v) > 0.6 else '#1e1e4a', fontweight='bold' if (i, j) in hi else 'normal')
        if (i, j) in hi:
            ax.add_patch(plt.Rectangle((j - .5, i - 1.5), 1, 1, fill=False, ec='#1e1e4a', lw=2.5))
ax.set_xticks(range(n - 1), labels[:-1], rotation=45, ha='right', fontsize=14, color='#1e1e4a')
ax.set_yticks(range(n - 1), labels[1:], fontsize=14, color='#1e1e4a')
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)
plt.tight_layout()
plt.savefig('/mnt/project-files/eda_final/figures/f15_슬라이드_상관히트맵.png', facecolor='white')
print(np.round(c[1, 0], 3), np.round(c[6, 3], 3), np.round(c[4, 3], 3))
