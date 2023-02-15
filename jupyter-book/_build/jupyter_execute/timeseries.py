#!/usr/bin/env python
# coding: utf-8

# (sec:timeseries)=
# # 시계열 데이터 

# In[1]:


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


# ## 날짜와 시간

# ### 날짜: `np.datetime64` 자료형

# In[2]:


np.datetime64('today')


# In[3]:


today = str(np.datetime64('today'))
today


# In[4]:


np.datetime64(today)


# In[5]:


np.datetime64('today') >= np.datetime64('2023-02-14')


# 년-월 형식도 가능하다.

# In[6]:


np.datetime64('2023-02')


# 년-월 형식에 날짜를 표현하면 1일로 지정된다.

# In[7]:


np.datetime64('2023-02', 'D')


# 년-월-일 에서 월 또는 1일 생략되면 항상 1월, 1일에 해당한다.

# In[8]:


np.datetime64('2005') == np.datetime64('2005-01-01')


# `datetime64` 어레이로 지정하는 방식은 다음과 같다.

# In[9]:


np.array(['2007-07-13', '2006-01-13', '2010-08-13'], dtype='datetime64')


# 년, 월, 주, 일 단위를 나타내는 기호는 다음과 같다.
# 
# | 코드 | 의미 |
# | :---: | :---: |
# | Y | 년 |
# | M | 월 |
# | W | 주 |
# | D | 일 |

# ### 시간: `np.datetime64` 자료형

# 시간을 시, 분, 초, 밀리초, 마이크로초, 나노초 등의 단위로 표현할 수 있다. 

# - 2023년 3월 2일 11시 30분

# In[10]:


np.datetime64('2023-03-02T11:30')


# - 2023년 3월 2일 21시 20분 6초 (저녁 9시 20분 6초)

# In[11]:


np.datetime64('2023-03-02T21:20:06')


# `T` 대신에 공백을 이용해도 된다.

# In[12]:


np.datetime64('2023-03-02 11:30')


# In[13]:


np.datetime64('2023-03-02 21:20:06')


# 시, 분, 초 단위를 나타내는 기호는 다음과 같다.
# 
# | 코드 | 의미 |
# | :---: | :---: |
# | h | 시|
# | m | 분 |
# | s | 초 |
# | ms | 밀리초| 
# | us | 마이크로초| 
# | ns | 나노초| 
# 
# **참고:** 
# 
# - 밀리초(millisecond): 천 분의 1초
# - 마이크로초(microsecond): 백만 분의 1초
# - 나노초(nanosecond): 10억 분의 1초

# ### NaT: Not a Time

# 날짜/시간이 아닌 것을 나타내는 값으로 `NaT` 를 사용한다.

# In[14]:


np.datetime64('NaT')


# NaT 을 대소문자 구분하지 않아도 모두 Nat로 인식된다.

# In[15]:


np.datetime64('nat')


# In[16]:


np.datetime64('nAT')


# `np.isnat()` 함수는 `Nat` 일 때 참을 반환한다.
# 어레이에 대해서는 항목별로 적용된다.

# In[17]:


np.isnat(np.array(["NaT", "2016-01-01"], dtype="datetime64[D]"))


# ### 날짜와 시간의 구간

# `np.arange()` 함수와 함께 사용하면 구간에 포함된 날짜로 어레이를 생성한다.

# - 일 단위 구간

# In[18]:


np.arange('2023-02', '2023-03', dtype='datetime64[D]')


# - 주 단위 구간

# In[19]:


np.arange('2023-02', '2023-05', dtype='datetime64[W]')


# - 월 단위 구간

# In[20]:


np.arange('2023-02', '2023-05', dtype='datetime64[M]')


# - 년 단위 구간

# In[21]:


np.arange('2023-02', '2032-03', dtype='datetime64[Y]')


# ### 시간의 크기: `np.timedelta64` 자료형

# 시간의 크기를 나타내는 자료형이다.

# - 하루의 시간

# In[22]:


np.timedelta64(1, 'D')


# * 4시간

# In[23]:


np.timedelta64(4, 'h')


# ### 날짜와 시간의 연산

# 날짜, 시간, 시간의 크기에 대해 덧셈, 뺄셈, 나눗셈 연산이 가능하다.

# - 시간의 연산은 일(day) 단위로 계산된다.
# - 2023년 3월 1일과 2022년 3월 1일의 시간 차이는 365일(1년)이다.

# In[24]:


np.datetime64('2023-03-01') - np.datetime64('2022-03-01')


# - 2023년 1월 1일에 20일을 더하면, 즉, 20일 후는 2023년 1월 21일이다.

# In[25]:


np.datetime64('2023') + np.timedelta64(20, 'D')


# - 2023년 6월 15일, 즉 2023년 6월 15일 0시 0분 0초에 12초를 더하면 2023년 6월 15일 0시 0분 12초를 가리킨다.

# In[26]:


np.datetime64('2023-06-15') + np.timedelta64(12, 's')


# - 2023년 6월 15일 0시 0분 12시간을 더하면 2023년 6월 15일 12시 0분을 가리킨다.

# In[27]:


np.datetime64('2023-06-15T00:00') + np.timedelta64(12, 'h')


# - 시간의 크기 나눗셈 결과는 부동소수점
# - 일주일(7일) 나누기 이틀(2일)는 3.5

# In[28]:


np.timedelta64(1,'W') / np.timedelta64(2,'D')


# - 나눗셈의 나머지 계산은 `np.timedelta64` 자료형
# - 일주일(7일)을 10일로 나눈 나머지는 7일

# In[29]:


np.timedelta64(1,'W') % np.timedelta64(10,'D')


# - Nat와의 연산은 무조건 NaT!

# In[30]:


np.datetime64('nat') - np.datetime64('2009-01-01')


# In[31]:


np.datetime64('2009-01-01') + np.timedelta64('nat')


# ## `datetime` 자료형

# In[32]:


from datetime import datetime
now = datetime.now()


# In[33]:


now


# In[34]:


print(now.year, now.month, now.day)


# In[35]:


delta = datetime(2011, 1, 7) - datetime(2008, 6, 24, 8, 15)
delta


# In[36]:


delta.days


# In[37]:


delta.seconds


# In[38]:


from datetime import timedelta


# In[39]:


start = datetime(2011, 1, 7)


# In[40]:


start + timedelta(12)


# In[41]:


start - 2 * timedelta(12)


# In[42]:


stamp = datetime(2011, 1, 3)


# In[43]:


str(stamp)


# In[44]:


stamp.strftime("%Y-%m-%d")


# In[45]:


value = "2011-01-03"
datetime.strptime(value, "%Y-%m-%d")


# In[46]:


datestrs = ["7/6/2011", "8/6/2011"]
[datetime.strptime(x, "%m/%d/%Y") for x in datestrs]


# `pd.to_datetime()` 함수

# In[47]:


datestrs = ["2011-07-06 12:00:00", "2011-08-06 00:00:00"]
pd.to_datetime(datestrs)


# In[48]:


idx = pd.to_datetime(datestrs + [None])
idx


# In[49]:


idx[2]


# In[50]:


pd.isna(idx)


# In[51]:


dates = [datetime(2011, 1, 2), datetime(2011, 1, 5),
         datetime(2011, 1, 7), datetime(2011, 1, 8),
         datetime(2011, 1, 10), datetime(2011, 1, 12)]


# In[52]:


ts = pd.Series(np.random.standard_normal(6), index=dates)
ts


# In[53]:


ts.index


# In[54]:


ts + ts[::2]


# In[55]:


ts.index


# In[56]:


ts.index.dtype


# In[57]:


stamp = ts.index[0]
stamp


# In[58]:


stamp = ts.index[2]
ts[stamp]


# In[59]:


ts["2011-01-10"]


# In[60]:


longer_ts = pd.Series(np.random.standard_normal(1000),
                      index=pd.date_range("2000-01-01", periods=1000))
longer_ts


# In[61]:


longer_ts["2001"]


# In[62]:


longer_ts["2001-05"]


# In[63]:


ts[datetime(2011, 1, 7):]


# In[64]:


ts[datetime(2011, 1, 7):datetime(2011, 1, 10)]


# In[65]:


ts


# In[66]:


ts["2011-01-06":"2011-01-11"]


# In[67]:


ts.truncate(after="2011-01-09")


# In[68]:


dates = pd.date_range("2000-01-01", periods=100, freq="W-WED")
long_df = pd.DataFrame(np.random.standard_normal((100, 4)),
                       index=dates,
                       columns=["Colorado", "Texas",
                                "New York", "Ohio"])


# In[69]:


long_df.loc["2001-05"]


# In[70]:


dates = pd.DatetimeIndex(["2000-01-01", "2000-01-02", "2000-01-02",
                          "2000-01-02", "2000-01-03"])
dup_ts = pd.Series(np.arange(5), index=dates)
dup_ts


# In[71]:


dup_ts.index.is_unique


# In[72]:


dup_ts["2000-01-03"]  # not duplicated


# In[73]:


dup_ts["2000-01-02"]  # duplicated


# In[74]:


grouped = dup_ts.groupby(level=0)


# In[75]:


grouped.mean()


# In[76]:


grouped.count()


# In[77]:


ts


# In[78]:


resampler = ts.resample("D")
resampler


# In[79]:


index = pd.date_range("2012-04-01", "2012-06-01")
index


# In[80]:


pd.date_range(start="2012-04-01", periods=20)


# In[81]:


pd.date_range(end="2012-06-01", periods=20)


# In[82]:


pd.date_range("2000-01-01", "2000-12-01", freq="BM")


# In[83]:


pd.date_range("2012-05-02 12:56:31", periods=5)


# In[84]:


pd.date_range("2012-05-02 12:56:31", periods=5, normalize=True)


# In[85]:


from pandas.tseries.offsets import Hour, Minute


# In[86]:


hour = Hour()
hour


# In[87]:


four_hours = Hour(4)
four_hours


# In[88]:


pd.date_range("2000-01-01", "2000-01-03 23:59", freq="4H")


# In[89]:


Hour(2) + Minute(30)


# In[90]:


pd.date_range("2000-01-01", periods=10, freq="1h30min")


# In[91]:


monthly_dates = pd.date_range("2012-01-01", "2012-09-01", freq="WOM-3FRI")
list(monthly_dates)


# In[92]:


ts = pd.Series(np.random.standard_normal(4),
               index=pd.date_range("2000-01-01", periods=4, freq="M"))
ts


# In[93]:


ts.shift(2)


# In[94]:


ts.shift(-2)


# In[95]:


ts.shift(2, freq="M")


# In[96]:


ts.shift(3, freq="D")


# In[97]:


ts.shift(1, freq="90T")


# In[98]:


from pandas.tseries.offsets import Day, MonthEnd


# In[99]:


now = datetime(2011, 11, 17)
now + 3 * Day()


# In[100]:


now + MonthEnd()


# In[101]:


now + MonthEnd(2)


# In[102]:


offset = MonthEnd()
offset.rollforward(now)


# In[103]:


offset.rollback(now)


# In[104]:


ts = pd.Series(np.random.standard_normal(20),
               index=pd.date_range("2000-01-15", periods=20, freq="4D"))
ts


# In[105]:


ts.groupby(MonthEnd().rollforward).mean()


# In[106]:


ts.resample("M").mean()


# In[107]:


import pytz

pytz.common_timezones[-5:]


# In[108]:


tz = pytz.timezone("America/New_York")
tz


# In[109]:


dates = pd.date_range("2012-03-09 09:30", periods=6)
ts = pd.Series(np.random.standard_normal(len(dates)), index=dates)

ts


# In[110]:


print(ts.index.tz)


# In[111]:


pd.date_range("2012-03-09 09:30", periods=10, tz="UTC")


# In[112]:


ts


# In[113]:


ts_utc = ts.tz_localize("UTC")
ts_utc


# In[114]:


ts_utc.index


# In[115]:


ts_utc.tz_convert("America/New_York")


# In[116]:


ts_eastern = ts.tz_localize("America/New_York")
ts_eastern.tz_convert("UTC")


# In[117]:


ts_eastern.tz_convert("Europe/Berlin")


# In[118]:


ts.index.tz_localize("Asia/Shanghai")


# In[119]:


stamp = pd.Timestamp("2011-03-12 04:00")
stamp_utc = stamp.tz_localize("utc")
stamp_utc.tz_convert("America/New_York")


# In[120]:


stamp_moscow = pd.Timestamp("2011-03-12 04:00", tz="Europe/Moscow")
stamp_moscow


# In[121]:


stamp_utc.value


# In[122]:


stamp_utc.tz_convert("America/New_York").value


# In[123]:


stamp = pd.Timestamp("2012-03-11 01:30", tz="US/Eastern")
stamp


# In[124]:


stamp + Hour()


# In[125]:


stamp = pd.Timestamp("2012-11-04 00:30", tz="US/Eastern")
stamp


# In[126]:


stamp + 2 * Hour()


# In[127]:


dates = pd.date_range("2012-03-07 09:30", periods=10, freq="B")
ts = pd.Series(np.random.standard_normal(len(dates)), index=dates)
ts


# In[128]:


ts1 = ts[:7].tz_localize("Europe/London")
ts2 = ts1[2:].tz_convert("Europe/Moscow")
result = ts1 + ts2
result.index


# In[129]:


p = pd.Period("2011", freq="A-DEC")
p


# In[130]:


p + 5


# In[131]:


p - 2


# In[132]:


pd.Period("2014", freq="A-DEC") - p


# In[133]:


periods = pd.period_range("2000-01-01", "2000-06-30", freq="M")
periods


# In[134]:


pd.Series(np.random.standard_normal(6), index=periods)


# In[135]:


values = ["2001Q3", "2002Q2", "2003Q1"]
index = pd.PeriodIndex(values, freq="Q-DEC")
index


# In[136]:


p = pd.Period("2011", freq="A-DEC")
p


# In[137]:


p.asfreq("M", how="start")


# In[138]:


p.asfreq("M", how="end")


# In[139]:


p.asfreq("M")


# In[140]:


p = pd.Period("2011", freq="A-JUN")
p


# In[141]:


p.asfreq("M", how="start")


# In[142]:


p.asfreq("M", how="end")


# In[143]:


p = pd.Period("Aug-2011", "M")
p.asfreq("A-JUN")


# In[144]:


periods = pd.period_range("2006", "2009", freq="A-DEC")
ts = pd.Series(np.random.standard_normal(len(periods)), index=periods)
ts


# In[145]:


ts.asfreq("M", how="start")


# In[146]:


ts.asfreq("B", how="end")


# In[147]:


p = pd.Period("2012Q4", freq="Q-JAN")
p


# In[148]:


p.asfreq("D", how="start")


# In[149]:


p.asfreq("D", how="end")


# In[150]:


p4pm = (p.asfreq("B", how="end") - 1).asfreq("T", how="start") + 16 * 60
p4pm


# In[151]:


p4pm.to_timestamp()


# In[152]:


periods = pd.period_range("2011Q3", "2012Q4", freq="Q-JAN")
ts = pd.Series(np.arange(len(periods)), index=periods)
ts


# In[153]:


new_periods = (periods.asfreq("B", "end") - 1).asfreq("H", "start") + 16
ts.index = new_periods.to_timestamp()
ts


# In[154]:


dates = pd.date_range("2000-01-01", periods=3, freq="M")
ts = pd.Series(np.random.standard_normal(3), index=dates)
ts


# In[155]:


pts = ts.to_period()
pts


# In[156]:


dates = pd.date_range("2000-01-29", periods=6)
ts2 = pd.Series(np.random.standard_normal(6), index=dates)
ts2


# In[157]:


ts2.to_period("M")


# In[158]:


pts = ts2.to_period()
pts


# In[159]:


pts.to_timestamp(how="end")


# In[160]:


data = pd.read_csv("examples/macrodata.csv")
data.head(5)


# In[161]:


data["year"]


# In[162]:


data["quarter"]


# In[163]:


index = pd.PeriodIndex(year=data["year"], quarter=data["quarter"],
                       freq="Q-DEC")
index


# In[164]:


data.index = index
data["infl"]


# In[165]:


dates = pd.date_range("2000-01-01", periods=100)
ts = pd.Series(np.random.standard_normal(len(dates)), index=dates)
ts


# In[166]:


ts.resample("M").mean()


# In[167]:


ts.resample("M", kind="period").mean()


# In[168]:


dates = pd.date_range("2000-01-01", periods=12, freq="T")
ts = pd.Series(np.arange(len(dates)), index=dates)
ts


# In[169]:


ts.resample("5min").sum()


# In[170]:


ts.resample("5min", closed="right").sum()


# In[171]:


ts.resample("5min", closed="right", label="right").sum()


# In[172]:


from pandas.tseries.frequencies import to_offset


# In[173]:


result = ts.resample("5min", closed="right", label="right").sum()
result.index = result.index + to_offset("-1s")
result


# In[174]:


ts = pd.Series(np.random.permutation(np.arange(len(dates))), index=dates)
ts.resample("5min").ohlc()


# In[175]:


frame = pd.DataFrame(np.random.standard_normal((2, 4)),
                     index=pd.date_range("2000-01-01", periods=2,
                                         freq="W-WED"),
                     columns=["Colorado", "Texas", "New York", "Ohio"])
frame


# In[176]:


df_daily = frame.resample("D").asfreq()
df_daily


# In[177]:


frame.resample("D").ffill()


# In[178]:


frame.resample("D").ffill(limit=2)


# In[179]:


frame.resample("W-THU").ffill()


# In[180]:


frame = pd.DataFrame(np.random.standard_normal((24, 4)),
                     index=pd.period_range("1-2000", "12-2001",
                                           freq="M"),
                     columns=["Colorado", "Texas", "New York", "Ohio"])
frame.head()


# In[181]:


annual_frame = frame.resample("A-DEC").mean()
annual_frame


# In[182]:


# Q-DEC: Quarterly, year ending in December
annual_frame.resample("Q-DEC").ffill()


# In[183]:


annual_frame.resample("Q-DEC", convention="end").asfreq()


# In[184]:


annual_frame.resample("Q-MAR").ffill()


# In[185]:


N = 15
times = pd.date_range("2017-05-20 00:00", freq="1min", periods=N)
df = pd.DataFrame({"time": times,
                   "value": np.arange(N)})
df


# In[186]:


df.set_index("time").resample("5min").count()


# In[187]:


df2 = pd.DataFrame({"time": times.repeat(3),
                    "key": np.tile(["a", "b", "c"], N),
                    "value": np.arange(N * 3.)})
df2.head(7)


# In[188]:


time_key = pd.Grouper(freq="5min")


# In[189]:


resampled = (df2.set_index("time")
             .groupby(["key", time_key])
             .sum())
resampled


# In[190]:


resampled.reset_index()


# In[191]:


close_px_all = pd.read_csv("examples/stock_px.csv",
                           parse_dates=True, index_col=0)
close_px = close_px_all[["AAPL", "MSFT", "XOM"]]
close_px = close_px.resample("B").ffill()


# In[192]:


close_px["AAPL"].plot()


# In[193]:


#! figure,id=apple_daily_ma250,title="Apple price with 250-day moving average"
close_px["AAPL"].rolling(250).mean().plot()


# In[194]:


plt.figure()
std250 = close_px["AAPL"].pct_change().rolling(250, min_periods=10).std()
std250[5:12]
#! figure,id=apple_daily_std250,title="Apple 250-day daily return standard deviation"
std250.plot()


# In[195]:


expanding_mean = std250.expanding().mean()


# In[196]:


plt.figure()
plt.style.use('grayscale')
#! figure,id=stocks_daily_ma60,title="Stock prices 60-day moving average (log y-axis)"
close_px.rolling(60).mean().plot(logy=True)


# In[197]:


close_px.rolling("20D").mean()


# In[198]:


plt.figure()

aapl_px = close_px["AAPL"]["2006":"2007"]

ma30 = aapl_px.rolling(30, min_periods=20).mean()
ewma30 = aapl_px.ewm(span=30).mean()

aapl_px.plot(style="k-", label="Price")
ma30.plot(style="k--", label="Simple Moving Avg")
ewma30.plot(style="k-", label="EW MA")
#! figure,id=timeseries_ewma,title="Simple moving average versus exponentially weighted"
plt.legend()


# In[199]:


plt.figure()

spx_px = close_px_all["SPX"]
spx_rets = spx_px.pct_change()
returns = close_px.pct_change()

corr = returns["AAPL"].rolling(125, min_periods=100).corr(spx_rets)
#! figure,id=roll_correl_aapl,title="Six-month AAPL return correlation to S&P 500"
corr.plot()


# In[200]:


plt.figure()

corr = returns.rolling(125, min_periods=100).corr(spx_rets)
#! figure,id=roll_correl_all,title="Six-month return correlations to S&P 500"
corr.plot()


# In[201]:


plt.figure()

from scipy.stats import percentileofscore
def score_at_2percent(x):
    return percentileofscore(x, 0.02)

result = returns["AAPL"].rolling(250).apply(score_at_2percent)
#! figure,id=roll_apply_ex,title="Percentile rank of 2% AAPL return over one-year window"
result.plot()

