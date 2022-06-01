#!/usr/bin/env python
# coding: utf-8

# (ch:seaborn)=
# # 데이터 시각화: seaborn

# - [Seaborn 홈페이지](https://seaborn.pydata.org/index.html)
# - [Seaborn intro](https://seaborn.pydata.org/introduction.html)
# - [Seaborn relational](https://seaborn.pydata.org/tutorial/relational.html)

# ## Intro

# In[1]:


# Import seaborn
import seaborn as sns

# Apply the default theme
sns.set_theme()

# Load an example dataset
tips = sns.load_dataset("tips")

# Create a visualization
sns.relplot(
    data=tips,
    x="total_bill", y="tip", col="time",
    hue="smoker", style="smoker", size="size",
)


# In[2]:


dots = sns.load_dataset("dots")
sns.relplot(
    data=dots, kind="line",
    x="time", y="firing_rate", col="align",
    hue="choice", size="coherence", style="choice",
    facet_kws=dict(sharex=False),
)


# In[3]:


fmri = sns.load_dataset("fmri")
sns.relplot(
    data=fmri, kind="line",
    x="timepoint", y="signal", col="region",
    hue="event", style="event",
)


# In[4]:


sns.lmplot(data=tips, x="total_bill", y="tip", col="time", hue="smoker")


# In[5]:


sns.displot(data=tips, x="total_bill", col="time", kde=True)


# In[6]:


sns.displot(data=tips, kind="ecdf", x="total_bill", col="time", hue="smoker", rug=True)


# In[7]:


sns.catplot(data=tips, kind="swarm", x="day", y="total_bill", hue="smoker")


# In[8]:


sns.catplot(data=tips, kind="violin", x="day", y="total_bill", hue="smoker", split=True)


# In[9]:


sns.catplot(data=tips, kind="bar", x="day", y="total_bill", hue="smoker")


# In[10]:


penguins = sns.load_dataset("penguins")
sns.jointplot(data=penguins, x="flipper_length_mm", y="bill_length_mm", hue="species")


# In[11]:


sns.pairplot(data=penguins, hue="species")


# In[12]:


g = sns.PairGrid(penguins, hue="species", corner=True)
g.map_lower(sns.kdeplot, hue=None, levels=5, color=".2")
g.map_lower(sns.scatterplot, marker="+")
g.map_diag(sns.histplot, element="step", linewidth=0, kde=True)
g.add_legend(frameon=True)
g.legend.set_bbox_to_anchor((.61, .6))


# In[13]:


sns.relplot(
    data=penguins,
    x="bill_length_mm", y="bill_depth_mm", hue="body_mass_g"
)


# In[14]:


sns.set_theme(style="ticks", font_scale=1.25)
g = sns.relplot(
    data=penguins,
    x="bill_length_mm", y="bill_depth_mm", hue="body_mass_g",
    palette="crest", marker="x", s=100,
)
g.set_axis_labels("Bill length (mm)", "Bill depth (mm)", labelpad=10)
g.legend.set_title("Body mass (g)")
g.figure.set_size_inches(6.5, 4.5)
g.ax.margins(.15)
g.despine(trim=True)


# In[ ]:




