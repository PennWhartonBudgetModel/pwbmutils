"""Setup script for package.
"""

from setuptools import setup
import os
import matplotlib

setup(
    name="pwbmutils",
    version="0.82",
    description="Collection of cross-component utility functions",
    url="https://github.com/PennWhartonBudgetModel/Utilities",
    author="Penn Wharton Budget Model",
    packages=["pwbmutils"],
    zip_safe=False,
    test_suite="nose.collector",
    install_requires=["luigi", "pandas", "matplotlib", "portalocker"],
    test_requires=["nose"]
)

style_guide = """
lines.linewidth: 4
lines.solid_capstyle: butt

axes.prop_cycle: cycler('color', ['004785', 'd7bc6a', 'a90533', 'b2b6a7', '532a85', 'c5093b', 'eeedea', '282f85', '026cb5', 'a8204e'])
axes.facecolor: FFFFFF
axes.labelsize: large
axes.axisbelow: true
axes.grid: true
axes.grid.axis: y
axes.edgecolor: FFFFFF
axes.linewidth: 2.0
axes.titlesize: x-large

patch.edgecolor: FFFFFF
patch.linewidth: 0.5

svg.fonttype: path

grid.linestyle: -
grid.linewidth: 1.0
grid.color: cbcbcb

xtick.major.size: 0
xtick.minor.size: 0
ytick.major.size: 0
ytick.minor.size: 0

font.size: 14.0
font.sans-serif: DejaVu Sans

savefig.edgecolor: FFFFFF
savefig.facecolor: FFFFFF

figure.subplot.left: 0.08
figure.subplot.right: 0.95
figure.subplot.bottom: 0.07
figure.facecolor: FFFFFF

legend.loc: best
legend.edgecolor: b2b6a7
legend.borderpad: 1
legend.frameon: true
"""

if not os.path.exists(os.path.join(matplotlib.get_configdir(), "stylelib")):
	os.makedirs(os.path.join(matplotlib.get_configdir(), "stylelib"))

with open(os.path.join(matplotlib.get_configdir(), "stylelib", "pwbm.mplstyle"), 'w') as file_out:
	file_out.writelines(style_guide)
