#!/usr/bin/env python
# coding: utf-8

# (sec:aggregation)=
# # 데이터 집계와 그룹 생성

# In[1]:


#! ipython suppress id=91bf744c97594b79aeaf6a9c84176193
get_ipython().run_line_magic('pushd', 'book-materials')
import numpy as np
import pandas as pd
PREVIOUS_MAX_ROWS = pd.options.display.max_rows
pd.options.display.max_columns = 20
pd.options.display.max_rows = 20
pd.options.display.max_colwidth = 80
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc("figure", figsize=(10, 6))
np.set_printoptions(precision=4, suppress=True)


# In[2]:


#! ipython id=a66384a336da4f5e80d115439d896670
import numpy as np
import pandas as pd


# In[3]:


#! ipython id=717f2535751f48a68d977477a3898589
df = pd.DataFrame({"key1" : ["a", "a", None, "b", "b", "a", None],
                   "key2" : pd.Series([1, 2, 1, 2, 1, None, 1], dtype="Int64"),
                   "data1" : np.random.standard_normal(7),
                   "data2" : np.random.standard_normal(7)})
df


# In[4]:


#! ipython id=24486d99929647449d6208aed5a966ef
grouped = df["data1"].groupby(df["key1"])
grouped


# In[5]:


#! ipython id=c29f2ae78cd746838c919a3b3d7cb9b0
grouped.mean()


# In[6]:


#! ipython id=b2f13da1d1db405db2714ce25d08b045
means = df["data1"].groupby([df["key1"], df["key2"]]).mean()
means


# In[7]:


#! ipython id=f6d3464627b6440ab9b6ab5267778ec0
means.unstack()


# In[8]:


#! ipython id=2b0bf87cf09644ee9f1f8ba9d4f2fb10
states = np.array(["OH", "CA", "CA", "OH", "OH", "CA", "OH"])
years = [2005, 2005, 2006, 2005, 2006, 2005, 2006]
df["data1"].groupby([states, years]).mean()


# In[9]:


#! ipython id=3eace3f65719478aa4e1fcc2002b04c9
df.groupby("key1").mean()
df.groupby("key2").mean()
df.groupby(["key1", "key2"]).mean()


# In[10]:


#! ipython id=b1a71909137c4bf5a1dadcdc297efc0f
df.groupby(["key1", "key2"]).size()


# In[11]:


#! ipython id=8C7C817AB41E4D5EB6D4E2D8C0FA31EF
df.groupby("key1", dropna=False).size()
df.groupby(["key1", "key2"], dropna=False).size()


# In[12]:


#! ipython id=ac652b305fef4edda286f9fa9c23f53b
df.groupby("key1").count()


# In[13]:


#! ipython id=37c961ad9c074ba1a2165ce95aabd63c
#! blockstart
for name, group in df.groupby("key1"):
    print(name)
    print(group)
#! blockend


# In[14]:


#! ipython id=8ec5c5d0eb0d431ba238881770479a40
#! blockstart
for (k1, k2), group in df.groupby(["key1", "key2"]):
    print((k1, k2))
    print(group)
#! blockend


# In[15]:


#! ipython id=dda859371f47444aab503b4e07658be1
pieces = {name: group for name, group in df.groupby("key1")}
pieces["b"]


# In[16]:


#! ipython id=b7d6ba2adfc84814b7fd09e6fa274ef4
grouped = df.groupby({"key1": "key", "key2": "key",
                      "data1": "data", "data2": "data"}, axis="columns")


# In[17]:


#! ipython id=ce2f6ff046f44cad8bafbf15a19a2cd5
#! blockstart
for group_key, group_values in grouped:
    print(group_key)
    print(group_values)
#! blockend


# In[18]:


#! ipython id=548bdde71e8f4741976a2cc620254fb9
df.groupby(["key1", "key2"])[["data2"]].mean()


# In[19]:


#! ipython id=ba1813aa18c9456db659b05313612c1b
s_grouped = df.groupby(["key1", "key2"])["data2"]
s_grouped
s_grouped.mean()


# In[20]:


#! ipython id=78aee68b5f504ff89e6698bd9bbec2b2
people = pd.DataFrame(np.random.standard_normal((5, 5)),
                      columns=["a", "b", "c", "d", "e"],
                      index=["Joe", "Steve", "Wanda", "Jill", "Trey"])
people.iloc[2:3, [1, 2]] = np.nan # Add a few NA values
people


# In[21]:


#! ipython id=4ef4621d217e4c70bdf9c6ea024b6af2
mapping = {"a": "red", "b": "red", "c": "blue",
           "d": "blue", "e": "red", "f" : "orange"}


# In[22]:


#! ipython id=dc0ed8d771324592b997cbb5742ae718
by_column = people.groupby(mapping, axis="columns")
by_column.sum()


# In[23]:


#! ipython id=1cc8cb1b67df408a8bc530c8cef2bf41
map_series = pd.Series(mapping)
map_series
people.groupby(map_series, axis="columns").count()


# In[24]:


#! ipython id=45ccbe40ecfe4d29864e091288af57e3
people.groupby(len).sum()


# In[25]:


#! ipython id=153a6214e5864c5186a8a1d52323c4f8
key_list = ["one", "one", "one", "two", "two"]
people.groupby([len, key_list]).min()


# In[26]:


#! ipython id=81eb2063b2ae4d7caecf1704255f893c
columns = pd.MultiIndex.from_arrays([["US", "US", "US", "JP", "JP"],
                                    [1, 3, 5, 1, 3]],
                                    names=["cty", "tenor"])
hier_df = pd.DataFrame(np.random.standard_normal((4, 5)), columns=columns)
hier_df


# In[27]:


#! ipython id=4d648f35e1f8461a96cad485d324551e
hier_df.groupby(level="cty", axis="columns").count()


# In[28]:


#! ipython id=643d0ed7414d41f8b5a32867b2539412
df
grouped = df.groupby("key1")
grouped["data1"].nsmallest(2)


# In[29]:


#! ipython id=cc3562f5cfeb4a14920643783ecadb69
def peak_to_peak(arr):
    return arr.max() - arr.min()
grouped.agg(peak_to_peak)


# In[30]:


#! ipython id=47959d0bf07b4c4298f729e6af6a3c76
grouped.describe()


# In[31]:


#! ipython id=ae1e841c962344d6a4ea9cc65da459f4
tips = pd.read_csv("examples/tips.csv")
tips.head()


# In[32]:


#! ipython id=212326c8b5954ee0b1a9993ba2eeb8d7
tips["tip_pct"] = tips["tip"] / tips["total_bill"]
tips.head()


# In[33]:


#! ipython id=b5d4697e5fe84be9993099e33b568345
grouped = tips.groupby(["day", "smoker"])


# In[34]:


#! ipython id=21d0430d3c794474a8a113cc10c0a59e
grouped_pct = grouped["tip_pct"]
grouped_pct.agg("mean")


# In[35]:


#! ipython id=2203ebe00b9345738e2c1748aeda7873
grouped_pct.agg(["mean", "std", peak_to_peak])


# In[36]:


#! ipython id=331922d6e0b8439fbba2d79df35ae3cb
grouped_pct.agg([("average", "mean"), ("stdev", np.std)])


# In[37]:


#! ipython id=b23e35d113f247f5b4ba6c7b57286cce
functions = ["count", "mean", "max"]
result = grouped[["tip_pct", "total_bill"]].agg(functions)
result


# In[38]:


#! ipython id=48aeddf0d8614a9daa16851bf8292777
result["tip_pct"]


# In[39]:


#! ipython id=696328eaf2a94a9bb379aedc13ece6f0
ftuples = [("Average", "mean"), ("Variance", np.var)]
grouped[["tip_pct", "total_bill"]].agg(ftuples)


# In[40]:


#! ipython id=9216a59f6365466ab7f76fa04a569de3
grouped.agg({"tip" : np.max, "size" : "sum"})
grouped.agg({"tip_pct" : ["min", "max", "mean", "std"],
             "size" : "sum"})


# In[41]:


#! ipython id=97bee90edc124121bde0d922a7da360d
tips.groupby(["day", "smoker"], as_index=False).mean()


# In[42]:


#! ipython id=b74e5bd1d36148bf86a6ae9fb8ac78d6
def top(df, n=5, column="tip_pct"):
    return df.sort_values(column, ascending=False)[:n]
top(tips, n=6)


# In[43]:


#! ipython id=afe083ef07204911ae57dfddc2a784a1
tips.groupby("smoker").apply(top)


# In[44]:


#! ipython id=0983c697950348288a7c7b9fb5cfdb4b
tips.groupby(["smoker", "day"]).apply(top, n=1, column="total_bill")


# In[45]:


#! ipython id=04225aab0dd44b12bbac2961aa35254b
result = tips.groupby("smoker")["tip_pct"].describe()
result
result.unstack("smoker")


# In[46]:


#! ipython id=c35103d700ee4488a481eb898f95001b
tips.groupby("smoker", group_keys=False).apply(top)


# In[47]:


#! ipython id=b9e64e9337f247c0a408ea0c7a9961c5
frame = pd.DataFrame({"data1": np.random.standard_normal(1000),
                      "data2": np.random.standard_normal(1000)})
frame.head()
quartiles = pd.cut(frame["data1"], 4)
quartiles.head(10)


# In[48]:


#! ipython id=ce0b3030e230497c929e7f36b6d4ddb9
def get_stats(group):
    return pd.DataFrame(
        {"min": group.min(), "max": group.max(),
        "count": group.count(), "mean": group.mean()}
    )

grouped = frame.groupby(quartiles)
grouped.apply(get_stats)


# In[49]:


#! ipython id=32b18bf4409a4743ac83db5969bc6006
grouped.agg(["min", "max", "count", "mean"])


# In[50]:


#! ipython id=acb41e66e6624f45a52c701f7188b2a9
quartiles_samp = pd.qcut(frame["data1"], 4, labels=False)
quartiles_samp.head()
grouped = frame.groupby(quartiles_samp)
grouped.apply(get_stats)


# In[51]:


#! ipython id=a5f76377f0184195828c907897eb39d1
s = pd.Series(np.random.standard_normal(6))
s[::2] = np.nan
s
s.fillna(s.mean())


# In[52]:


#! ipython id=b4f45efcbdd0491f906ca6a4847bab1e
states = ["Ohio", "New York", "Vermont", "Florida",
          "Oregon", "Nevada", "California", "Idaho"]
group_key = ["East", "East", "East", "East",
             "West", "West", "West", "West"]
data = pd.Series(np.random.standard_normal(8), index=states)
data


# In[53]:


#! ipython id=72f2656d298f413f8720b68b62b51674
data[["Vermont", "Nevada", "Idaho"]] = np.nan
data
data.groupby(group_key).size()
data.groupby(group_key).count()
data.groupby(group_key).mean()


# In[54]:


#! ipython id=dfc55860e6ea449aadb99677fc0e2986
def fill_mean(group):
    return group.fillna(group.mean())

data.groupby(group_key).apply(fill_mean)


# In[55]:


#! ipython id=4abfc8bfccfd41ea8d64ac98443dac45
fill_values = {"East": 0.5, "West": -1}
def fill_func(group):
    return group.fillna(fill_values[group.name])

data.groupby(group_key).apply(fill_func)


# In[56]:


#! ipython verbatim id=05397b351c8e498c947d802a1cb48677
suits = ["H", "S", "C", "D"]  # Hearts, Spades, Clubs, Diamonds
card_val = (list(range(1, 11)) + [10] * 3) * 4
base_names = ["A"] + list(range(2, 11)) + ["J", "K", "Q"]
cards = []
for suit in suits:
    cards.extend(str(num) + suit for num in base_names)

deck = pd.Series(card_val, index=cards)


# In[57]:


#! ipython id=d54e45a5b95f4e84ab900beaeffc032e
deck.head(13)


# In[58]:


#! ipython id=1793b875d18f4a9c8eaf378979a65b8e
def draw(deck, n=5):
    return deck.sample(n)
draw(deck)


# In[59]:


#! ipython id=5f9eadad6e6b4033a68f0cb222e87013
def get_suit(card):
    # last letter is suit
    return card[-1]

deck.groupby(get_suit).apply(draw, n=2)


# In[60]:


#! ipython id=56e0c862e6984555a30a33d3800ccf1e
deck.groupby(get_suit, group_keys=False).apply(draw, n=2)


# In[61]:


#! ipython id=2f6291892da840ff9f2cc04ddc89a1a0
df = pd.DataFrame({"category": ["a", "a", "a", "a",
                                "b", "b", "b", "b"],
                   "data": np.random.standard_normal(8),
                   "weights": np.random.uniform(size=8)})
df


# In[62]:


#! ipython id=f3785264a084456e81b4fe42320ecd65
grouped = df.groupby("category")
def get_wavg(group):
    return np.average(group["data"], weights=group["weights"])

grouped.apply(get_wavg)


# In[63]:


#! ipython id=1113b4b05f2b495c9fa94c42165ac7a3
close_px = pd.read_csv("examples/stock_px.csv", parse_dates=True,
                       index_col=0)
close_px.info()
close_px.tail(4)


# In[64]:


#! ipython id=78ce3a267cfc491e8301e301f8a57342
def spx_corr(group):
    return group.corrwith(group["SPX"])


# In[65]:


#! ipython id=840cfbfdd33d4b82ba7013ea91095473
rets = close_px.pct_change().dropna()


# In[66]:


#! ipython id=bf0ac26ec88548f9b9bb5e7b5a3e6248
def get_year(x):
    return x.year

by_year = rets.groupby(get_year)
by_year.apply(spx_corr)


# In[67]:


#! ipython id=6dd546477c5548819f3610f8d0ff8987
def corr_aapl_msft(group):
    return group["AAPL"].corr(group["MSFT"])
by_year.apply(corr_aapl_msft)


# In[68]:


#! ipython verbatim id=64e6e813b0e948cd8cfa443fcbe485ad
import statsmodels.api as sm
def regress(data, yvar=None, xvars=None):
    Y = data[yvar]
    X = data[xvars]
    X["intercept"] = 1.
    result = sm.OLS(Y, X).fit()
    return result.params


# In[69]:


#! ipython id=78fdb9df1f984f3cb2ae8ae664af69f9
by_year.apply(regress, yvar="AAPL", xvars=["SPX"])


# In[70]:


#! ipython id=9d6f36a2555f44f29e689cd6a86c6c8b
df = pd.DataFrame({'key': ['a', 'b', 'c'] * 4,
                   'value': np.arange(12.)})
df


# In[71]:


#! ipython id=4f22990eed5f4794981b426d9690a067
g = df.groupby('key')['value']
g.mean()


# In[72]:


#! ipython id=e3dfd39e3cfb4b45a4d4fd3a0df8ba35
def get_mean(group):
    return group.mean()
g.transform(get_mean)


# In[73]:


#! ipython id=df3dbb405f0a418cb8f38b1d693126dc
g.transform('mean')


# In[74]:


#! ipython id=7c617bcf7d4c442abda7927077f4501a
def times_two(group):
    return group * 2
g.transform(times_two)


# In[75]:


#! ipython id=f90592bf16b7442097b9e22a099e1c75
def get_ranks(group):
    return group.rank(ascending=False)
g.transform(get_ranks)


# In[76]:


#! ipython id=8997dec2bb2e44978c3d39e98d12bec1
def normalize(x):
    return (x - x.mean()) / x.std()


# In[77]:


#! ipython id=d06c11e725f744abb0b0309eb2c04526
g.transform(normalize)
g.apply(normalize)


# In[78]:


#! ipython id=36f1cd239f9a4b218ea0d3dc86776890
g.transform('mean')
normalized = (df['value'] - g.transform('mean')) / g.transform('std')
normalized


# In[79]:


#! ipython id=2502077543534742bebd6ec88eec4d04
tips.head()
tips.pivot_table(index=["day", "smoker"])


# In[80]:


#! ipython id=5f3d264eb32842e39cca60645b498611
tips.pivot_table(index=["time", "day"], columns="smoker",
                 values=["tip_pct", "size"])


# In[81]:


#! ipython id=f65b89a55539440cac1405a620ed2758
tips.pivot_table(index=["time", "day"], columns="smoker",
                 values=["tip_pct", "size"], margins=True)


# In[82]:


#! ipython id=b597770eae7542cfb95aa7168d4293a8
tips.pivot_table(index=["time", "smoker"], columns="day",
                 values="tip_pct", aggfunc=len, margins=True)


# In[83]:


#! ipython id=307e306fd97449aa96c00b4c1af2887b
tips.pivot_table(index=["time", "size", "smoker"], columns="day",
                 values="tip_pct", fill_value=0)


# In[84]:


#! ipython id=c4ea7c59067e4ed5acce365201cec661
from io import StringIO
#! blockstart
data = """Sample  Nationality  Handedness
1   USA  Right-handed
2   Japan    Left-handed
3   USA  Right-handed
4   Japan    Right-handed
5   Japan    Left-handed
6   Japan    Right-handed
7   USA  Right-handed
8   USA  Left-handed
9   Japan    Right-handed
10  USA  Right-handed"""
#! blockend
data = pd.read_table(StringIO(data), sep="\s+")


# In[85]:


#! ipython id=d7d271d116954e7d8237cabcbc227c52
data


# In[86]:


#! ipython id=64e891aa9bc94090b29654b1f44dc426
pd.crosstab(data["Nationality"], data["Handedness"], margins=True)


# In[87]:


#! ipython id=5d0f91726eb748ecaa904c3c703c69cb
pd.crosstab([tips["time"], tips["day"]], tips["smoker"], margins=True)


# In[88]:


#! ipython suppress id=231271ac720a43deadca4fb4fce4b133
get_ipython().run_line_magic('popd', '')


# In[89]:


#! ipython suppress id=553ffce9df1b4513bff1925e8c5e6c50
pd.options.display.max_rows = PREVIOUS_MAX_ROWS

