#!/usr/bin/env python
# coding: utf-8

# (sec:combination)=
# # 데이터 결합: merge-join-concat

# In[1]:


import numpy as np
import pandas as pd
pd.options.display.max_rows = 20
pd.options.display.max_colwidth = 80
pd.options.display.max_columns = 20
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc("figure", figsize=(10, 6))
np.set_printoptions(precision=4, suppress=True)


# In[2]:


data = pd.Series(np.random.uniform(size=9),
                 index=[["a", "a", "a", "b", "b", "c", "c", "d", "d"],
                        [1, 2, 3, 1, 3, 1, 2, 2, 3]])
data


# In[3]:


data.index


# In[4]:


data["b"]
data["b":"c"]
data.loc[["b", "d"]]


# In[5]:


data.loc[:, 2]


# In[6]:


data.unstack()


# In[7]:


data.unstack().stack()


# In[8]:


frame = pd.DataFrame(np.arange(12).reshape((4, 3)),
                     index=[["a", "a", "b", "b"], [1, 2, 1, 2]],
                     columns=[["Ohio", "Ohio", "Colorado"],
                              ["Green", "Red", "Green"]])
frame


# In[9]:


frame.index.names = ["key1", "key2"]
frame.columns.names = ["state", "color"]
frame


# In[10]:


frame.index.nlevels


# In[11]:


frame["Ohio"]


# In[12]:


frame.swaplevel("key1", "key2")


# In[13]:


frame.sort_index(level=1)
frame.swaplevel(0, 1).sort_index(level=0)


# In[14]:


frame.groupby(level="key2").sum()
frame.groupby(level="color", axis="columns").sum()


# In[15]:


frame = pd.DataFrame({"a": range(7), "b": range(7, 0, -1),
                      "c": ["one", "one", "one", "two", "two",
                            "two", "two"],
                      "d": [0, 1, 2, 0, 1, 2, 3]})
frame


# In[16]:


frame2 = frame.set_index(["c", "d"])
frame2


# In[17]:


frame.set_index(["c", "d"], drop=False)


# In[18]:


frame2.reset_index()


# In[19]:


df1 = pd.DataFrame({"key": ["b", "b", "a", "c", "a", "a", "b"],
                    "data1": pd.Series(range(7), dtype="Int64")})
df2 = pd.DataFrame({"key": ["a", "b", "d"],
                    "data2": pd.Series(range(3), dtype="Int64")})
df1
df2


# In[20]:


pd.merge(df1, df2)


# In[21]:


pd.merge(df1, df2, on="key")


# In[22]:


df3 = pd.DataFrame({"lkey": ["b", "b", "a", "c", "a", "a", "b"],
                    "data1": pd.Series(range(7), dtype="Int64")})
df4 = pd.DataFrame({"rkey": ["a", "b", "d"],
                    "data2": pd.Series(range(3), dtype="Int64")})
pd.merge(df3, df4, left_on="lkey", right_on="rkey")


# In[23]:


pd.merge(df1, df2, how="outer")
pd.merge(df3, df4, left_on="lkey", right_on="rkey", how="outer")


# In[24]:


df1 = pd.DataFrame({"key": ["b", "b", "a", "c", "a", "b"],
                    "data1": pd.Series(range(6), dtype="Int64")})
df2 = pd.DataFrame({"key": ["a", "b", "a", "b", "d"],
                    "data2": pd.Series(range(5), dtype="Int64")})
df1
df2
pd.merge(df1, df2, on="key", how="left")


# In[25]:


pd.merge(df1, df2, how="inner")


# In[26]:


left = pd.DataFrame({"key1": ["foo", "foo", "bar"],
                     "key2": ["one", "two", "one"],
                     "lval": pd.Series([1, 2, 3], dtype='Int64')})
right = pd.DataFrame({"key1": ["foo", "foo", "bar", "bar"],
                      "key2": ["one", "one", "one", "two"],
                      "rval": pd.Series([4, 5, 6, 7], dtype='Int64')})
pd.merge(left, right, on=["key1", "key2"], how="outer")


# In[27]:


pd.merge(left, right, on="key1")


# In[28]:


pd.merge(left, right, on="key1", suffixes=("_left", "_right"))


# In[29]:


left1 = pd.DataFrame({"key": ["a", "b", "a", "a", "b", "c"],
                      "value": pd.Series(range(6), dtype="Int64")})
right1 = pd.DataFrame({"group_val": [3.5, 7]}, index=["a", "b"])
left1
right1
pd.merge(left1, right1, left_on="key", right_index=True)


# In[30]:


pd.merge(left1, right1, left_on="key", right_index=True, how="outer")


# In[31]:


lefth = pd.DataFrame({"key1": ["Ohio", "Ohio", "Ohio",
                               "Nevada", "Nevada"],
                      "key2": [2000, 2001, 2002, 2001, 2002],
                      "data": pd.Series(range(5), dtype="Int64")})
righth_index = pd.MultiIndex.from_arrays(
    [
        ["Nevada", "Nevada", "Ohio", "Ohio", "Ohio", "Ohio"],
        [2001, 2000, 2000, 2000, 2001, 2002]
    ]
)
righth = pd.DataFrame({"event1": pd.Series([0, 2, 4, 6, 8, 10], dtype="Int64",
                                           index=righth_index),
                       "event2": pd.Series([1, 3, 5, 7, 9, 11], dtype="Int64",
                                           index=righth_index)})
lefth
righth


# In[32]:


pd.merge(lefth, righth, left_on=["key1", "key2"], right_index=True)
pd.merge(lefth, righth, left_on=["key1", "key2"],
         right_index=True, how="outer")


# In[33]:


left2 = pd.DataFrame([[1., 2.], [3., 4.], [5., 6.]],
                     index=["a", "c", "e"],
                     columns=["Ohio", "Nevada"]).astype("Int64")
right2 = pd.DataFrame([[7., 8.], [9., 10.], [11., 12.], [13, 14]],
                      index=["b", "c", "d", "e"],
                      columns=["Missouri", "Alabama"]).astype("Int64")
left2
right2
pd.merge(left2, right2, how="outer", left_index=True, right_index=True)


# In[34]:


left2.join(right2, how="outer")


# In[35]:


left1.join(right1, on="key")


# In[36]:


another = pd.DataFrame([[7., 8.], [9., 10.], [11., 12.], [16., 17.]],
                       index=["a", "c", "e", "f"],
                       columns=["New York", "Oregon"])
another
left2.join([right2, another])
left2.join([right2, another], how="outer")


# In[37]:


arr = np.arange(12).reshape((3, 4))
arr
np.concatenate([arr, arr], axis=1)


# In[38]:


s1 = pd.Series([0, 1], index=["a", "b"], dtype="Int64")
s2 = pd.Series([2, 3, 4], index=["c", "d", "e"], dtype="Int64")
s3 = pd.Series([5, 6], index=["f", "g"], dtype="Int64")


# In[39]:


s1
s2
s3
pd.concat([s1, s2, s3])


# In[40]:


pd.concat([s1, s2, s3], axis="columns")


# In[41]:


s4 = pd.concat([s1, s3])
s4
pd.concat([s1, s4], axis="columns")
pd.concat([s1, s4], axis="columns", join="inner")


# In[42]:


result = pd.concat([s1, s1, s3], keys=["one", "two", "three"])
result
result.unstack()


# In[43]:


pd.concat([s1, s2, s3], axis="columns", keys=["one", "two", "three"])


# In[44]:


df1 = pd.DataFrame(np.arange(6).reshape(3, 2), index=["a", "b", "c"],
                   columns=["one", "two"])
df2 = pd.DataFrame(5 + np.arange(4).reshape(2, 2), index=["a", "c"],
                   columns=["three", "four"])
df1
df2
pd.concat([df1, df2], axis="columns", keys=["level1", "level2"])


# In[45]:


pd.concat({"level1": df1, "level2": df2}, axis="columns")


# In[46]:


pd.concat([df1, df2], axis="columns", keys=["level1", "level2"],
          names=["upper", "lower"])


# In[47]:


df1 = pd.DataFrame(np.random.standard_normal((3, 4)),
                   columns=["a", "b", "c", "d"])
df2 = pd.DataFrame(np.random.standard_normal((2, 3)),
                   columns=["b", "d", "a"])
df1
df2


# In[48]:


pd.concat([df1, df2], ignore_index=True)


# In[49]:


a = pd.Series([np.nan, 2.5, 0.0, 3.5, 4.5, np.nan],
              index=["f", "e", "d", "c", "b", "a"])
b = pd.Series([0., np.nan, 2., np.nan, np.nan, 5.],
              index=["a", "b", "c", "d", "e", "f"])
a
b
np.where(pd.isna(a), b, a)


# In[50]:


a.combine_first(b)


# In[51]:


df1 = pd.DataFrame({"a": [1., np.nan, 5., np.nan],
                    "b": [np.nan, 2., np.nan, 6.],
                    "c": range(2, 18, 4)})
df2 = pd.DataFrame({"a": [5., 4., np.nan, 3., 7.],
                    "b": [np.nan, 3., 4., 6., 8.]})
df1
df2
df1.combine_first(df2)


# In[52]:


data = pd.DataFrame(np.arange(6).reshape((2, 3)),
                    index=pd.Index(["Ohio", "Colorado"], name="state"),
                    columns=pd.Index(["one", "two", "three"],
                    name="number"))
data


# In[53]:


result = data.stack()
result


# In[54]:


result.unstack()


# In[55]:


result.unstack(level=0)
result.unstack(level="state")


# In[56]:


s1 = pd.Series([0, 1, 2, 3], index=["a", "b", "c", "d"], dtype="Int64")
s2 = pd.Series([4, 5, 6], index=["c", "d", "e"], dtype="Int64")
data2 = pd.concat([s1, s2], keys=["one", "two"])
data2


# In[57]:


data2.unstack()
data2.unstack().stack()
data2.unstack().stack(dropna=False)


# In[58]:


df = pd.DataFrame({"left": result, "right": result + 5},
                  columns=pd.Index(["left", "right"], name="side"))
df
df.unstack(level="state")


# In[59]:


df.unstack(level="state").stack(level="side")


# In[60]:


data = pd.read_csv("examples/macrodata.csv")
data = data.loc[:, ["year", "quarter", "realgdp", "infl", "unemp"]]
data.head()


# In[61]:


periods = pd.PeriodIndex(year=data.pop("year"),
                         quarter=data.pop("quarter"),
                         name="date")
periods
data.index = periods.to_timestamp("D")
data.head()


# In[62]:


data = data.reindex(columns=["realgdp", "infl", "unemp"])
data.columns.name = "item"
data.head()


# In[63]:


long_data = (data.stack()
             .reset_index()
             .rename(columns={0: "value"}))


# In[64]:


long_data[:10]


# In[65]:


pivoted = long_data.pivot(index="date", columns="item",
                          values="value")
pivoted.head()


# In[66]:


long_data.index.name = None


# In[67]:


long_data["value2"] = np.random.standard_normal(len(long_data))
long_data[:10]


# In[68]:


pivoted = long_data.pivot(index="date", columns="item")
pivoted.head()
pivoted["value"].head()


# In[69]:


unstacked = long_data.set_index(["date", "item"]).unstack(level="item")
unstacked.head()


# In[70]:





# In[70]:


df = pd.DataFrame({"key": ["foo", "bar", "baz"],
                   "A": [1, 2, 3],
                   "B": [4, 5, 6],
                   "C": [7, 8, 9]})
df


# In[71]:


melted = pd.melt(df, id_vars="key")
melted


# In[72]:


reshaped = melted.pivot(index="key", columns="variable",
                        values="value")
reshaped


# In[73]:


reshaped.reset_index()


# In[74]:


pd.melt(df, id_vars="key", value_vars=["A", "B"])


# In[75]:


pd.melt(df, value_vars=["A", "B", "C"])
pd.melt(df, value_vars=["key", "A", "B"])

