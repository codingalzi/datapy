#!/usr/bin/env python
# coding: utf-8

# (sec:timeseries)=
# # 시계열 데이터 

# In[1]:


#! ipython suppress id=1b8eaf74bf6e455ea8440642e23a1745
get_ipython().run_line_magic('pushd', 'book-materials')
import numpy as np
import pandas as pd
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc("figure", figsize=(10, 6))
PREVIOUS_MAX_ROWS = pd.options.display.max_rows
pd.options.display.max_columns = 20
pd.options.display.max_rows = 20
pd.options.display.max_colwidth = 80
np.set_printoptions(precision=4, suppress=True)


# In[2]:


#! ipython id=b8f6434f6f9540249d053d94cee92435
import numpy as np
import pandas as pd


# ## 날짜와 시간

# In[3]:


np.datetime64('today')


# In[4]:


str(np.datetime64('today'))


# In[5]:


np.datetime64('2023-02-14')


# In[6]:


np.datetime64('today') == np.datetime64('2023-02-14')


# In[7]:


np.datetime64('today') >= np.datetime64('2023-02-14')


# In[8]:


np.datetime64('2023-02')


# In[9]:


np.datetime64('2023-02', 'D')


# In[10]:


np.array(['2007-07-13', '2006-01-13', '2010-08-13'], dtype='datetime64')


# | 코드 | 의미 |
# | :---: | :---: |
# | Y | 년 |
# | M | 월 |
# | W | 주 |
# | D | 일 |

# In[11]:


np.arange('2023-02', '2023-03', dtype='datetime64[D]')


# In[12]:


np.arange('2023-02', '2023-05', dtype='datetime64[W]')


# In[13]:


np.arange('2023-02', '2023-05', dtype='datetime64[M]')


# In[14]:


np.arange('2023-02', '2032-03', dtype='datetime64[Y]')


# In[15]:


np.datetime64('2005') == np.datetime64('2005-01-01')


# In[16]:


np.timedelta64(1, 'D')


# In[17]:


np.timedelta64(4, 'h')


# In[18]:


np.datetime64('2009-01-01') - np.datetime64('2008-01-01')


# In[19]:


np.datetime64('2009') + np.timedelta64(20, 'D')


# In[20]:


np.datetime64('2011-06-15T00:00') + np.timedelta64(12, 'h')


# In[21]:


np.timedelta64(1,'W') / np.timedelta64(1,'D')


# In[22]:


np.timedelta64(1,'W') % np.timedelta64(10,'D')


# In[23]:


np.datetime64('nat') - np.datetime64('2009-01-01')


# In[24]:


np.datetime64('2009-01-01') + np.timedelta64('nat')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[25]:


#! ipython id=77aa02db8c0740baa9d429ef9f9c92c3
from datetime import datetime
now = datetime.now()
now
now.year, now.month, now.day


# In[26]:


#! ipython id=7939b9e6813c4810af2eb7d69016ac97
delta = datetime(2011, 1, 7) - datetime(2008, 6, 24, 8, 15)
delta
delta.days
delta.seconds


# In[27]:


#! ipython id=62309078c64449508cf0353453566a3b
from datetime import timedelta
start = datetime(2011, 1, 7)
start + timedelta(12)
start - 2 * timedelta(12)


# In[28]:


#! ipython id=2e863437108e49619f828b3399a23675
stamp = datetime(2011, 1, 3)
str(stamp)
stamp.strftime("%Y-%m-%d")


# In[29]:


#! ipython id=504bb23dbed740c697961017072ba0a5
value = "2011-01-03"
datetime.strptime(value, "%Y-%m-%d")
datestrs = ["7/6/2011", "8/6/2011"]
[datetime.strptime(x, "%m/%d/%Y") for x in datestrs]


# In[30]:


#! ipython id=66f136014e5e477aba0e773ab3137d23
datestrs = ["2011-07-06 12:00:00", "2011-08-06 00:00:00"]
pd.to_datetime(datestrs)


# In[31]:


#! ipython id=ab30c892b1cd40ee981e793e194fe34e
idx = pd.to_datetime(datestrs + [None])
idx
idx[2]
pd.isna(idx)


# In[32]:


#! ipython id=c65a386b5ee44b77ae927a17a586fdce
dates = [datetime(2011, 1, 2), datetime(2011, 1, 5),
         datetime(2011, 1, 7), datetime(2011, 1, 8),
         datetime(2011, 1, 10), datetime(2011, 1, 12)]
ts = pd.Series(np.random.standard_normal(6), index=dates)
ts


# In[33]:


#! ipython id=b29f5a3e0b3744ac8d9d10b87b10a23d
ts.index


# In[34]:


#! ipython id=b345022b87e2494592f0fc50fdc3461a
ts + ts[::2]


# In[35]:


#! ipython id=713bb1b1f7424b228fc940b8ff21adcf
ts.index.dtype


# In[36]:


#! ipython id=1de1ab6f38e640c7ae177e117612248f
stamp = ts.index[0]
stamp


# In[37]:


#! ipython id=962135198c7243b9b891d420cf335e69
stamp = ts.index[2]
ts[stamp]


# In[38]:


#! ipython id=01e57a024aa849279a3ebacc160772ca
ts["2011-01-10"]


# In[39]:


#! ipython id=d499c6cab3f64469acf7942aaa7bc5f1
longer_ts = pd.Series(np.random.standard_normal(1000),
                      index=pd.date_range("2000-01-01", periods=1000))
longer_ts
longer_ts["2001"]


# In[40]:


#! ipython id=4fc49d1ae9fd4f0b8e1a69999896298c
longer_ts["2001-05"]


# In[41]:


#! ipython id=59f52116836440a88e46e01adf808d35
ts[datetime(2011, 1, 7):]
ts[datetime(2011, 1, 7):datetime(2011, 1, 10)]


# In[42]:


#! ipython id=484c72fd4b744f5bbcb8a6493b45018c
ts
ts["2011-01-06":"2011-01-11"]


# In[43]:


#! ipython id=a6db533c922e4684a8c267288880f20b
ts.truncate(after="2011-01-09")


# In[44]:


#! ipython id=7324548996ac413f81d5e7014af682a0
dates = pd.date_range("2000-01-01", periods=100, freq="W-WED")
long_df = pd.DataFrame(np.random.standard_normal((100, 4)),
                       index=dates,
                       columns=["Colorado", "Texas",
                                "New York", "Ohio"])
long_df.loc["2001-05"]


# In[45]:


#! ipython id=b64ec3dd408148cda95729e8c7bd9b47
dates = pd.DatetimeIndex(["2000-01-01", "2000-01-02", "2000-01-02",
                          "2000-01-02", "2000-01-03"])
dup_ts = pd.Series(np.arange(5), index=dates)
dup_ts


# In[46]:


#! ipython id=aafe9e9742b74098aac057bb029e620d
dup_ts.index.is_unique


# In[47]:


#! ipython id=1963a191266e4095a6ed630a8e606956
dup_ts["2000-01-03"]  # not duplicated
dup_ts["2000-01-02"]  # duplicated


# In[48]:


#! ipython id=7895e016504f421ab8bd9cafeaeffea6
grouped = dup_ts.groupby(level=0)
grouped.mean()
grouped.count()


# In[49]:


#! ipython id=215d6274cd3e4bc3bcacdd8e5ce9f838
ts
resampler = ts.resample("D")
resampler


# In[50]:


#! ipython id=c202d9e9284c4fd28200376310af6070
index = pd.date_range("2012-04-01", "2012-06-01")
index


# In[51]:


#! ipython id=19e8a1e8306f4ed2b5329228064c50a6
pd.date_range(start="2012-04-01", periods=20)
pd.date_range(end="2012-06-01", periods=20)


# In[52]:


#! ipython id=8c50f185b572469eb8caa8ca321df712
pd.date_range("2000-01-01", "2000-12-01", freq="BM")


# In[53]:


#! ipython id=aa62083a0254482486435a4e2d777694
pd.date_range("2012-05-02 12:56:31", periods=5)


# In[54]:


#! ipython id=2fa04c5942ae482d87ef4eba22d7dc24
pd.date_range("2012-05-02 12:56:31", periods=5, normalize=True)


# In[55]:


#! ipython id=4a7a44f7b0394f4a8f9fd7bc31aae08f
from pandas.tseries.offsets import Hour, Minute
hour = Hour()
hour


# In[56]:


#! ipython id=502bfa3bd2b340c7bc8bd5a6e0855c63
four_hours = Hour(4)
four_hours


# In[57]:


#! ipython id=e2b7e93362014fc0a72d5c0675bba1a7
pd.date_range("2000-01-01", "2000-01-03 23:59", freq="4H")


# In[58]:


#! ipython id=beef8c7807e34b1ea65756437085bab5
Hour(2) + Minute(30)


# In[59]:


#! ipython id=5bb25a584faa49718052115bd3b3ae39
pd.date_range("2000-01-01", periods=10, freq="1h30min")


# In[60]:


#! ipython id=db9f1fc3d9a5419994e1adda66c83212
monthly_dates = pd.date_range("2012-01-01", "2012-09-01", freq="WOM-3FRI")
list(monthly_dates)


# In[61]:


#! ipython id=b2397e270f56489e89f63a2c1732d8d5
ts = pd.Series(np.random.standard_normal(4),
               index=pd.date_range("2000-01-01", periods=4, freq="M"))
ts
ts.shift(2)
ts.shift(-2)


# In[62]:


#! ipython id=30cdaeba864a452b894ce82253efcec7
ts.shift(2, freq="M")


# In[63]:


#! ipython id=43fd4a3b864347008c9e58579d1318ec
ts.shift(3, freq="D")
ts.shift(1, freq="90T")


# In[64]:


#! ipython id=d3a295c2d66d4ee49501d8e52df4006c
from pandas.tseries.offsets import Day, MonthEnd
now = datetime(2011, 11, 17)
now + 3 * Day()


# In[65]:


#! ipython id=840a4a58d2444b42b7f71aa5f69af613
now + MonthEnd()
now + MonthEnd(2)


# In[66]:


#! ipython id=947ac8d4eae84bcb9abbd6250957070f
offset = MonthEnd()
offset.rollforward(now)
offset.rollback(now)


# In[67]:


#! ipython id=efd27228c8f54689b2e1e476204ba317
ts = pd.Series(np.random.standard_normal(20),
               index=pd.date_range("2000-01-15", periods=20, freq="4D"))
ts
ts.groupby(MonthEnd().rollforward).mean()


# In[68]:


#! ipython id=b6d6087bdfb4474f91a0cf774283f1d1
ts.resample("M").mean()


# In[69]:


#! ipython id=19c326a03d9a4cd783fbcc8cb5ff421c
import pytz
pytz.common_timezones[-5:]


# In[70]:


#! ipython id=81652670d9d84ed1bc65dff261cd0937
tz = pytz.timezone("America/New_York")
tz


# In[71]:


#! ipython id=27a1920372f0414cb335c9ac80c1a883
dates = pd.date_range("2012-03-09 09:30", periods=6)
ts = pd.Series(np.random.standard_normal(len(dates)), index=dates)
ts


# In[72]:


#! ipython id=fd6da4e7623d40babb80385bd6b80a36
print(ts.index.tz)


# In[73]:


#! ipython id=e03db4a0b6d34d66a88f0be927424a1b
pd.date_range("2012-03-09 09:30", periods=10, tz="UTC")


# In[74]:


#! ipython id=0dd2997562634b6c9161adc215855962
ts
ts_utc = ts.tz_localize("UTC")
ts_utc
ts_utc.index


# In[75]:


#! ipython id=49c15434eb0c47bf950d42390118f378
ts_utc.tz_convert("America/New_York")


# In[76]:


#! ipython id=01e4044008b14d7f9e4990a1e66210e3
ts_eastern = ts.tz_localize("America/New_York")
ts_eastern.tz_convert("UTC")
ts_eastern.tz_convert("Europe/Berlin")


# In[77]:


#! ipython id=1dcc4db3bd4b43138c22293079cafe68
ts.index.tz_localize("Asia/Shanghai")


# In[78]:


#! ipython id=e3618c9788f040a38fac2443dac25892
stamp = pd.Timestamp("2011-03-12 04:00")
stamp_utc = stamp.tz_localize("utc")
stamp_utc.tz_convert("America/New_York")


# In[79]:


#! ipython id=8715a838b21f4789a6b58e4ceb0a56e4
stamp_moscow = pd.Timestamp("2011-03-12 04:00", tz="Europe/Moscow")
stamp_moscow


# In[80]:


#! ipython id=0aed2bac3c41471899b081a582412f4e
stamp_utc.value
stamp_utc.tz_convert("America/New_York").value


# In[81]:


#! ipython id=b778152898374964a084de3b3b63f935
stamp = pd.Timestamp("2012-03-11 01:30", tz="US/Eastern")
stamp
stamp + Hour()


# In[82]:


#! ipython id=c0d18d84985d4ede88ec95727c7fb8e9
stamp = pd.Timestamp("2012-11-04 00:30", tz="US/Eastern")
stamp
stamp + 2 * Hour()


# In[83]:


#! ipython id=8cde92a79e444f44b5f952eabb4b7491
dates = pd.date_range("2012-03-07 09:30", periods=10, freq="B")
ts = pd.Series(np.random.standard_normal(len(dates)), index=dates)
ts
ts1 = ts[:7].tz_localize("Europe/London")
ts2 = ts1[2:].tz_convert("Europe/Moscow")
result = ts1 + ts2
result.index


# In[84]:


#! ipython id=bae9a7cce5cd496b8b3756412b670013
p = pd.Period("2011", freq="A-DEC")
p


# In[85]:


#! ipython id=518401c31bc8419bbf28956e2c9be595
p + 5
p - 2


# In[86]:


#! ipython id=e5ff985e90fe44579705c3d3cdf5e5b7
pd.Period("2014", freq="A-DEC") - p


# In[87]:


#! ipython id=ef000f892654403b9f476bb7e1d9bacd
periods = pd.period_range("2000-01-01", "2000-06-30", freq="M")
periods


# In[88]:


#! ipython id=d3426c6da33b4175891f9d8d4caa4ab8
pd.Series(np.random.standard_normal(6), index=periods)


# In[89]:


#! ipython id=47db0721ff4a4ca89bdc74daae703a55
values = ["2001Q3", "2002Q2", "2003Q1"]
index = pd.PeriodIndex(values, freq="Q-DEC")
index


# In[90]:


#! ipython id=348880d80dc14062bb80c374d953f7ce
p = pd.Period("2011", freq="A-DEC")
p
p.asfreq("M", how="start")
p.asfreq("M", how="end")
p.asfreq("M")


# In[91]:


#! ipython id=669af5f271964c7cbefe3265cde35c6c
p = pd.Period("2011", freq="A-JUN")
p
p.asfreq("M", how="start")
p.asfreq("M", how="end")


# In[92]:


#! ipython id=ec04cdb713554c0fa0b4162e372b9876
p = pd.Period("Aug-2011", "M")
p.asfreq("A-JUN")


# In[93]:


#! ipython id=979cb059424a48f6a887fb50fdb66adf
periods = pd.period_range("2006", "2009", freq="A-DEC")
ts = pd.Series(np.random.standard_normal(len(periods)), index=periods)
ts
ts.asfreq("M", how="start")


# In[94]:


#! ipython id=51ab6bd5e33e433a8786b0a3fe239dd2
ts.asfreq("B", how="end")


# In[95]:


#! ipython id=38b59581b62f4808a145d4b5bac7d04c
p = pd.Period("2012Q4", freq="Q-JAN")
p


# In[96]:


#! ipython id=03f4b2f382e84260bfccec8df35b5903
p.asfreq("D", how="start")
p.asfreq("D", how="end")


# In[97]:


#! ipython id=629d1fcb8b5d42329da5126f166a028f
p4pm = (p.asfreq("B", how="end") - 1).asfreq("T", how="start") + 16 * 60
p4pm
p4pm.to_timestamp()


# In[98]:


#! ipython id=2b671198cc014db3bf349878fab5e36a
periods = pd.period_range("2011Q3", "2012Q4", freq="Q-JAN")
ts = pd.Series(np.arange(len(periods)), index=periods)
ts
new_periods = (periods.asfreq("B", "end") - 1).asfreq("H", "start") + 16
ts.index = new_periods.to_timestamp()
ts


# In[99]:


#! ipython id=debf05cadb0f4a94a7174e2f8859f352
dates = pd.date_range("2000-01-01", periods=3, freq="M")
ts = pd.Series(np.random.standard_normal(3), index=dates)
ts
pts = ts.to_period()
pts


# In[100]:


#! ipython id=6ac8d11fe6a1408cb83be89895617dd6
dates = pd.date_range("2000-01-29", periods=6)
ts2 = pd.Series(np.random.standard_normal(6), index=dates)
ts2
ts2.to_period("M")


# In[101]:


#! ipython id=de1111bbae4a4343945546f43c1af706
pts = ts2.to_period()
pts
pts.to_timestamp(how="end")


# In[102]:


#! ipython id=d14658aeb9db49f090190ba82c255d4e
data = pd.read_csv("examples/macrodata.csv")
data.head(5)
data["year"]
data["quarter"]


# In[103]:


#! ipython id=4b1ab91f7f9d4f958b10ae49617ae6c6
index = pd.PeriodIndex(year=data["year"], quarter=data["quarter"],
                       freq="Q-DEC")
index
data.index = index
data["infl"]


# In[104]:


#! ipython id=07232ac59e2a49f18606b1b9422b2889
dates = pd.date_range("2000-01-01", periods=100)
ts = pd.Series(np.random.standard_normal(len(dates)), index=dates)
ts
ts.resample("M").mean()
ts.resample("M", kind="period").mean()


# In[105]:


#! ipython id=12dad7e46e2449eca9f2482399b82397
dates = pd.date_range("2000-01-01", periods=12, freq="T")
ts = pd.Series(np.arange(len(dates)), index=dates)
ts


# In[106]:


#! ipython id=b187a814d71c4a7bbb8c1c8750d9133f
ts.resample("5min").sum()


# In[107]:


#! ipython id=450f99fca60945bab78ddf104cedabf9
ts.resample("5min", closed="right").sum()


# In[108]:


#! ipython id=6587ba3145214dc8b3f50ccb40bc88b8
ts.resample("5min", closed="right", label="right").sum()


# In[109]:


#! ipython id=c48ee1ea64c2498e8ede907aea647016
from pandas.tseries.frequencies import to_offset
result = ts.resample("5min", closed="right", label="right").sum()
result.index = result.index + to_offset("-1s")
result


# In[110]:


#! ipython id=e036e195212b453985a4bc219cd308f3
ts = pd.Series(np.random.permutation(np.arange(len(dates))), index=dates)
ts.resample("5min").ohlc()


# In[111]:


#! ipython id=22d7658abb234e16a56b2637142c053c
frame = pd.DataFrame(np.random.standard_normal((2, 4)),
                     index=pd.date_range("2000-01-01", periods=2,
                                         freq="W-WED"),
                     columns=["Colorado", "Texas", "New York", "Ohio"])
frame


# In[112]:


#! ipython id=7a564b646cbc45d78d95f6333692fd84
df_daily = frame.resample("D").asfreq()
df_daily


# In[113]:


#! ipython id=ecb1c4fc9803419b849c214cddaddae3
frame.resample("D").ffill()


# In[114]:


#! ipython id=a84044fc3dbe4f0597b997ec51c96490
frame.resample("D").ffill(limit=2)


# In[115]:


#! ipython id=3f5e6bf2f6844db9ae9ac57ee806df7e
frame.resample("W-THU").ffill()


# In[116]:


#! ipython id=42a429ef95bc45fdb9c595f3b3ffd163
frame = pd.DataFrame(np.random.standard_normal((24, 4)),
                     index=pd.period_range("1-2000", "12-2001",
                                           freq="M"),
                     columns=["Colorado", "Texas", "New York", "Ohio"])
frame.head()
annual_frame = frame.resample("A-DEC").mean()
annual_frame


# In[117]:


#! ipython id=21bc509f9fc340b6882974f3ec17e715
# Q-DEC: Quarterly, year ending in December
annual_frame.resample("Q-DEC").ffill()
annual_frame.resample("Q-DEC", convention="end").asfreq()


# In[118]:


#! ipython id=9d82a3b714164b4dad4eceaeadeda604
annual_frame.resample("Q-MAR").ffill()


# In[119]:


#! ipython id=f23204097cbd44b3a899d2cbaa35c2bd
N = 15
times = pd.date_range("2017-05-20 00:00", freq="1min", periods=N)
df = pd.DataFrame({"time": times,
                   "value": np.arange(N)})
df


# In[120]:


#! ipython id=806fd8e5d2aa413f8c990b6acebde10d
df.set_index("time").resample("5min").count()


# In[121]:


#! ipython id=9a28095367094308ad46b129e69586aa
df2 = pd.DataFrame({"time": times.repeat(3),
                    "key": np.tile(["a", "b", "c"], N),
                    "value": np.arange(N * 3.)})
df2.head(7)


# In[122]:


#! ipython id=0991852576124da587038d8939d3de61
time_key = pd.Grouper(freq="5min")


# In[123]:


#! ipython id=a7c8d07161384a4385a4d82fffa4e7ae
resampled = (df2.set_index("time")
             .groupby(["key", time_key])
             .sum())
resampled
resampled.reset_index()


# In[124]:


#! ipython id=3ff50b0ceeef40c4bbe32b4b8cf3824a
close_px_all = pd.read_csv("examples/stock_px.csv",
                           parse_dates=True, index_col=0)
close_px = close_px_all[["AAPL", "MSFT", "XOM"]]
close_px = close_px.resample("B").ffill()


# In[125]:


#! ipython id=3a7907583c5c464eb2d146e8ddcb479f
close_px["AAPL"].plot()
#! figure,id=apple_daily_ma250,title="Apple price with 250-day moving average"
close_px["AAPL"].rolling(250).mean().plot()


# In[126]:


#! ipython id=cd2ea550f4ab44bebddcdef67fc7990b
plt.figure()
std250 = close_px["AAPL"].pct_change().rolling(250, min_periods=10).std()
std250[5:12]
#! figure,id=apple_daily_std250,title="Apple 250-day daily return standard deviation"
std250.plot()


# In[127]:


#! ipython id=504b4010407f4edcac81dfa106681206
expanding_mean = std250.expanding().mean()


# In[128]:


#! ipython suppress id=828830f0853b45388516d1d716305702
plt.figure()


# In[129]:


#! ipython id=757ccd62b4c64042add7a055b90d3f79
plt.style.use('grayscale')
#! figure,id=stocks_daily_ma60,title="Stock prices 60-day moving average (log y-axis)"
close_px.rolling(60).mean().plot(logy=True)


# In[130]:


#! ipython id=d1df05357bef4ef5be80524c35b9407a
close_px.rolling("20D").mean()


# In[131]:


#! ipython suppress id=f65d7f1f960c4466af77c7e80a9fdf4c
plt.figure()


# In[132]:


#! ipython id=fac9c661db8a4c77bd6e450f0ca0f082
aapl_px = close_px["AAPL"]["2006":"2007"]

ma30 = aapl_px.rolling(30, min_periods=20).mean()
ewma30 = aapl_px.ewm(span=30).mean()

aapl_px.plot(style="k-", label="Price")
ma30.plot(style="k--", label="Simple Moving Avg")
ewma30.plot(style="k-", label="EW MA")
#! figure,id=timeseries_ewma,title="Simple moving average versus exponentially weighted"
plt.legend()


# In[133]:


#! ipython suppress id=cf8d4580bade4701b8cc6c1b05c4ed97
plt.figure()


# In[134]:


#! ipython id=850496d7c94b423f994cbc710002ccf7
spx_px = close_px_all["SPX"]
spx_rets = spx_px.pct_change()
returns = close_px.pct_change()


# In[135]:


#! ipython id=f065c26f19e7491b8f573f8d0f0d67e3
corr = returns["AAPL"].rolling(125, min_periods=100).corr(spx_rets)
#! figure,id=roll_correl_aapl,title="Six-month AAPL return correlation to S&P 500"
corr.plot()


# In[136]:


#! ipython suppress id=f67e1433110f4a6a8e72034ead461e09
plt.figure()


# In[137]:


#! ipython id=e312071fe8a74319afc03f447030c468
corr = returns.rolling(125, min_periods=100).corr(spx_rets)
#! figure,id=roll_correl_all,title="Six-month return correlations to S&P 500"
corr.plot()


# In[138]:


#! ipython suppress id=e0f7372085924896960706add4fa5f56
plt.figure()


# In[139]:


#! ipython id=e176480d4d034f1bbaa1b4c09caf9df6
from scipy.stats import percentileofscore
def score_at_2percent(x):
    return percentileofscore(x, 0.02)

result = returns["AAPL"].rolling(250).apply(score_at_2percent)
#! figure,id=roll_apply_ex,title="Percentile rank of 2% AAPL return over one-year window"
result.plot()


# In[140]:


#! ipython suppress id=419d9badc37c41888f19b0fd158061dd
get_ipython().run_line_magic('popd', '')


# In[141]:


#! ipython suppress id=1ff3016eeceb4d3e9a95f134c7ab9512
pd.options.display.max_rows = PREVIOUS_MAX_ROWS

