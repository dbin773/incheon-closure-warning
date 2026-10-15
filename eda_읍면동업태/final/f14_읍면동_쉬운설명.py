# 읍면동_ vs 읍면동업태_ 쉬운 설명 그림 (부평동 예시 + 음식점 행 산점도)
import pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl
mpl.rcParams['font.family'] = 'WenQuanYi Zen Hei'
mpl.rcParams['axes.unicode_minus'] = False
BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e4e3df'
d = pd.read_csv('/mnt/project-files/eda/인천_분기x읍면동x업태_다음분기폐업률.csv')

ex = d[(d.기준분기 == '2026Q2') & (d.읍면동_개편후 == '부평동')].set_index('업태구분명')
r, c = ex.loc['음식점'], ex.loc['카페']

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={'width_ratios': [1.25, 1]})
fig.patch.set_facecolor('white')

# A: 점포수 막대
n_r, n_c, n_all = r.읍면동업태_영업점포수, c.읍면동업태_영업점포수, r.읍면동_영업점포수
a1.barh(1, n_r, color=BLUE, height=0.5)
a1.barh(1, n_c, left=n_r + 8, color=ORANGE, height=0.5)
a1.barh(0, n_all, color=MUTED, height=0.5)
a1.text(n_r / 2, 1, f'음식점 {n_r:,.0f}곳', ha='center', va='center', color='white', fontsize=13)
a1.text(n_r + 8 + n_c + 30, 1, f'카페 {n_c:,.0f}곳', ha='left', va='center', color=INK, fontsize=13)
a1.text(n_all / 2, 0, f'동네 전체 {n_all:,.0f}곳', ha='center', va='center', color='white', fontsize=13)
a1.set_yticks([1, 0], ['읍면동업태_\n(업태별)', '읍면동_\n(동네 전체)'], fontsize=12, color=INK)
a1.set_xlim(0, n_all * 1.25); a1.set_xticks([])
for s in a1.spines.values(): s.set_visible(False)
a1.set_title('① 예시: 부평동 2026년 2분기 점포수\n동네 전체 = 음식점 + 카페, 그중 88%가 음식점', loc='left', fontsize=14, color=INK)
a1.text(0, -0.75, f"폐업률도 비슷하다:  음식점 {r.읍면동업태_폐업률:.1%}  vs  동네 전체 {r.읍면동_폐업률:.1%}",
        fontsize=13, color=INK, transform=a1.transData)

# B: 음식점 행 산점도 (학습 2023~2025, 점포 30개 이상)
t = d[(d.업태구분명 == '음식점') & (d.기준연도 <= 2025) & (d.읍면동업태_영업점포수 >= 30)]
a2.plot([0, 0.08], [0, 0.08], color=MUTED, lw=1, ls='--')
a2.scatter(t.읍면동업태_폐업률, t.읍면동_폐업률, s=14, color=BLUE, alpha=0.45, edgecolor='none')
a2.set_xlim(0, 0.08); a2.set_ylim(0, 0.08)
fmt = mpl.ticker.PercentFormatter(1, decimals=0)
a2.xaxis.set_major_formatter(fmt); a2.yaxis.set_major_formatter(fmt)
a2.set_xlabel('음식점만의 폐업률 (읍면동업태_)', fontsize=12, color=INK)
a2.set_ylabel('동네 전체 폐업률 (읍면동_)', fontsize=12, color=INK)
a2.grid(color=GRID, lw=0.8); a2.set_axisbelow(True)
for s in ['top', 'right']: a2.spines[s].set_visible(False)
a2.text(0.079, 0.006, '점선 = 두 숫자가 똑같은 곳', fontsize=12, color=MUTED, ha='right')
a2.set_title(f'② 모든 동네·분기 음식점 행 {len(t):,}개\n점이 거의 점선 위 → 같은 숫자를 두 번 넣는 셈', loc='left', fontsize=14, color=INK)

plt.tight_layout()
plt.savefig('/mnt/project-files/eda_final/figures/f14_읍면동_쉬운설명.png', dpi=150, bbox_inches='tight')
print(len(t), t[['읍면동업태_폐업률','읍면동_폐업률']].corr().iloc[0,1])
